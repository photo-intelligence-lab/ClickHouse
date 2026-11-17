#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>
#include <base/defines.h>
#include <Common/CacheLine.h>
#include <Common/getNumberOfCPUCoresToUse.h>

namespace DB
{

/// Priority increases recorded by cache hits and not yet applied to the priority queue.
/// A hit only stores a record here, and whoever next holds the queue write lock takes a batch,
/// so the write lock is taken once per batch of hits instead of once per hit.
template <typename Record>
class PendingPriorityIncreases
{
public:
    ~PendingPriorityIncreases() { delete stripes.load(std::memory_order_relaxed); }

    /// Allocates the memory for the records on the first call, so a queue without hits has none.
    /// Must be called before `tryAdd`.
    void createStripesIfNeeded()
    {
        if (stripes.load(std::memory_order_acquire))
            return;

        auto created = std::make_unique<Stripes>(std::max(1u, getNumberOfCPUCoresToUse()));
        Stripes * expected = nullptr;
        if (stripes.compare_exchange_strong(expected, created.get(), std::memory_order_acq_rel))
            created.release(); /// NOLINT(bugprone-unused-return-value)
    }

    enum class AddResult
    {
        Added,
        /// Another thread is using the stripe of the calling thread.
        Busy,
        Full,
    };

    AddResult tryAdd(Record && record)
    {
        auto & stripe = getStripeOfCurrentThread(*stripes.load(std::memory_order_acquire));
        if (!stripe.mutex.try_lock())
            return AddResult::Busy;
        std::lock_guard lock(stripe.mutex, std::adopt_lock);
        const size_t size = stripe.size.load(std::memory_order_relaxed);
        if (size == stripe.records.size())
            return AddResult::Full;
        stripe.records[size] = std::move(record);
        stripe.size.store(size + 1, std::memory_order_relaxed);
        return AddResult::Added;
    }

    /// Moves up to `records.size()` records out, to the front of `records`, and returns their number.
    /// Records of one thread keep their order.
    /// Calls must not overlap: each one starts from the stripe after the previous start.
    size_t take(std::span<Record> records)
    {
        auto * current_stripes = stripes.load(std::memory_order_acquire);
        if (!current_stripes)
            return 0;

        const size_t stripes_count = current_stripes->size();
        size_t taken = 0;
        for (size_t i = 0; i < stripes_count && taken < records.size(); ++i)
            taken += takeFromStripe((*current_stripes)[(first_stripe_to_take + i) % stripes_count], records.subspan(taken));
        first_stripe_to_take = (first_stripe_to_take + 1) % stripes_count;
        return taken;
    }

    /// Same, for the stripe of the calling thread only.
    size_t takeOfCurrentThread(std::span<Record> records)
    {
        auto * current_stripes = stripes.load(std::memory_order_acquire);
        return current_stripes ? takeFromStripe(getStripeOfCurrentThread(*current_stripes), records) : 0;
    }

private:
    static constexpr size_t max_records_per_stripe = 32;

    /// One per CPU core, so that hits of different threads rarely share a mutex or a cache line.
    struct alignas(CH_CACHE_LINE_SIZE) Stripe
    {
        std::mutex mutex;
        /// Atomic so that an empty stripe is skipped without taking its mutex.
        std::atomic<size_t> size = 0;
        std::array<Record, max_records_per_stripe> records TSA_GUARDED_BY(mutex);
    };
    using Stripes = std::vector<Stripe>;

    static Stripe & getStripeOfCurrentThread(Stripes & stripes_)
    {
        static std::atomic<size_t> threads_count = 0;
        static thread_local const size_t thread_index = threads_count.fetch_add(1, std::memory_order_relaxed);
        return stripes_[thread_index % stripes_.size()];
    }

    static size_t takeFromStripe(Stripe & stripe, std::span<Record> records)
    {
        static_assert(std::is_nothrow_move_assignable_v<Record>);

        if (stripe.size.load(std::memory_order_relaxed) == 0)
            return 0;

        std::lock_guard lock(stripe.mutex);
        const size_t size = stripe.size.load(std::memory_order_relaxed);
        const size_t count = std::min(size, records.size());
        std::move(stripe.records.begin(), stripe.records.begin() + count, records.begin());
        std::move(stripe.records.begin() + count, stripe.records.begin() + size, stripe.records.begin());
        stripe.size.store(size - count, std::memory_order_relaxed);
        return count;
    }

    std::atomic<Stripes *> stripes = nullptr;
    size_t first_stripe_to_take = 0;
};

}
