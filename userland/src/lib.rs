#![no_std]

extern crate alloc;

use alloc::borrow::Cow;
#[cfg(any(
    feature = "baseline_trace_vm",
    feature = "handoff_trace_vm",
    feature = "single_async_trace_vm",
    feature = "nested_spawn_trace_vm",
    feature = "multi_async_trace_vm",
    feature = "sleep_wakeup_trace_vm"
))]
use awkernel_async_lib::{scheduler::SchedulerType, task};

pub async fn main() -> Result<(), Cow<'static, str>> {
    #[cfg(any(
        feature = "baseline_trace_vm",
        feature = "handoff_trace_vm",
        feature = "single_async_trace_vm",
        feature = "nested_spawn_trace_vm",
        feature = "multi_async_trace_vm",
        feature = "sleep_wakeup_trace_vm"
    ))]
    {
        return Ok(());
    }

    #[cfg(not(any(
        feature = "baseline_trace_vm",
        feature = "handoff_trace_vm",
        feature = "single_async_trace_vm",
        feature = "nested_spawn_trace_vm",
        feature = "multi_async_trace_vm",
        feature = "sleep_wakeup_trace_vm"
    )))]
    {
        awkernel_services::run().await;

        #[cfg(feature = "rd_gen_to_dags")]
        rd_gen_to_dags::run().await; // run the rd_gen_to_dags application

        #[cfg(feature = "test_network")]
        test_network::run().await; // test for network

        #[cfg(feature = "test_pubsub")]
        test_pubsub::run().await; // test for pubsub

        #[cfg(feature = "test_rpi_hal")]
        test_rpi_hal::run().await; // test for RPi HAL

        #[cfg(feature = "test_graphics")]
        test_graphics::run().await; // test for graphics

        #[cfg(feature = "test_prioritized_fifo")]
        test_prioritized_fifo::run().await; // test for prioritized_fifo

        #[cfg(feature = "test_multi_prioritized_scheduler")]
        test_multi_prioritized_scheduler::run().await; // test for multi prioritized scheduler

        #[cfg(feature = "test_prioritized_rr")]
        test_prioritized_rr::run().await; // test for prioritized_rr

        #[cfg(feature = "test_async_mutex")]
        test_async_mutex::run().await; // test for async_mutex

        #[cfg(feature = "test_gedf")]
        test_gedf::run().await; // test for Global Earliest Deadline First scheduler

        #[cfg(feature = "test_measure_channel")]
        test_measure_channel::run().await; // measure channel

        #[cfg(feature = "test_measure_channel_heavy")]
        test_measure_channel_heavy::run().await; // measure channel heavy

        #[cfg(feature = "test_dag")]
        test_dag::run().await; // test for DAG

        #[cfg(feature = "test_dvfs")]
        test_dvfs::run().await; // test for DVFS

        #[cfg(feature = "test_voluntary_preemption")]
        test_voluntary_preemption::run().await; // test for voluntary preemption

        Ok(())
    }
}

#[cfg(feature = "baseline_trace_vm")]
pub fn install_baseline_trace_vm() {
    awkernel_async_lib::baseline_trace::reset();

    let task_id = task::spawn(
        "[Awkernel] baseline trace worker".into(),
        async { Ok(()) },
        SchedulerType::PrioritizedFIFO(31),
    );

    awkernel_async_lib::baseline_trace::arm_dump_on_complete(task_id);
}

#[cfg(feature = "handoff_trace_vm")]
pub fn install_handoff_trace_vm() {
    awkernel_async_lib::baseline_trace::reset();

    let task_id = task::spawn(
        "[Awkernel] handoff trace worker".into(),
        async { Ok(()) },
        SchedulerType::PrioritizedFIFO(31),
    );

    awkernel_async_lib::baseline_trace::arm_dump_on_complete(task_id);
}

#[cfg(feature = "single_async_trace_vm")]
pub fn install_single_async_trace_vm() {
    awkernel_async_lib::baseline_trace::reset();

    let task_id = task::spawn(
        "[Awkernel] single_async trace root".into(),
        async { trace_minimal::run_single_async().await },
        SchedulerType::PrioritizedFIFO(31),
    );

    awkernel_async_lib::baseline_trace::arm_dump_on_complete(task_id);
}

#[cfg(feature = "nested_spawn_trace_vm")]
pub fn install_nested_spawn_trace_vm() {
    awkernel_async_lib::baseline_trace::reset();

    let task_id = task::spawn(
        "[Awkernel] nested_spawn trace root".into(),
        async { trace_minimal::run_nested_spawn().await },
        SchedulerType::PrioritizedFIFO(31),
    );

    awkernel_async_lib::baseline_trace::arm_dump_on_complete(task_id);
}

#[cfg(feature = "multi_async_trace_vm")]
pub fn install_multi_async_trace_vm() {
    awkernel_async_lib::baseline_trace::reset();

    let task_id = task::spawn(
        "[Awkernel] multi_async trace root".into(),
        async { trace_minimal::run_multi_async().await },
        SchedulerType::PrioritizedFIFO(31),
    );

    awkernel_async_lib::baseline_trace::arm_dump_on_complete(task_id);
}

#[cfg(feature = "sleep_wakeup_trace_vm")]
pub fn install_sleep_wakeup_trace_vm() {
    awkernel_async_lib::baseline_trace::reset();

    let task_id = task::spawn(
        "[Awkernel] sleep_wakeup trace root".into(),
        async { trace_minimal::run_sleep_wakeup().await },
        SchedulerType::PrioritizedFIFO(31),
    );

    awkernel_async_lib::baseline_trace::arm_dump_on_complete(task_id);
}
