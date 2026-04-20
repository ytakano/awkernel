#![no_std]

#[macro_use]
extern crate alloc;

use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};
use awkernel_async_lib::{
    scheduler::SchedulerType,
    sleep,
    task::{self, TaskResult},
};
use awkernel_lib::{console, sync::mutex::MCSNode, IS_STD};
use blisp::{embedded, runtime::FFI};
use core::time::Duration;
use num_bigint::BigInt;
use num_traits::ToPrimitive;

const SERVICE_NAME: &str = "[Awkernel] shell";

pub fn init() {
    let task_id = task::spawn(
        SERVICE_NAME.into(),
        console_handler(),
        SchedulerType::PrioritizedFIFO(0),
    );

    if let Some(irq) = awkernel_lib::console::irq_id() {
        if awkernel_lib::interrupt::register_handler(
            irq,
            "serial port (awkernel_shell)".into(),
            Box::new(move |_irq| task::wake(task_id)),
        )
        .is_err()
        {
            log::warn!("Failed to initialize UART's interrupt handler.");
        }
    }
}

async fn console_handler() -> TaskResult {
    log::info!("Start {SERVICE_NAME}.");

    #[allow(unused_mut)]
    let mut functions: Vec<Box<dyn FFI + Send>> = vec![
        Box::new(HelpFfi),
        Box::new(TaskFfi),
        Box::new(InterruptFfi),
        Box::new(IfconfigFfi),
        Box::new(NetdumpFfi),
        Box::new(AddIpv4Ffi),
        Box::new(Arping4Ffi),
        Box::new(SetGateway4Ffi),
        Box::new(RebootFfi),
        Box::new(ShutdownFfi),
    ];

    #[cfg(feature = "perf")]
    functions.push(Box::new(PerfFfi));

    let code = if cfg!(feature = "perf") {
        format!("{CODE}\r\n{PERF_CODE}")
    } else {
        CODE.to_string()
    };

    let exprs = blisp::init(&code, functions).unwrap();
    let blisp_ctx = blisp::typing(exprs).unwrap();

    let mut line = Vec::new();

    console::print("\r\nWelcome to Awkernel!\r\n\r\n");
    console::print("You can use BLisp language as follows.\r\n");
    console::print("https://ytakano.github.io/blisp/\r\n\r\n");
    console::print("> (factorial 20)\r\n");
    console::print("2432902008176640000\r\n");
    console::print("> (+ 10 20)\r\n");
    console::print("30\r\n");

    console::print("\r\nEnjoy!\r\n\r\n");

    console::print("> ");
    loop {
        if let Some(c) = console::get() {
            if c == 0x08 || c == 0x7F || c == 0x15 {
                // backspace, delete
                if !line.is_empty() {
                    if !IS_STD {
                        console::put(0x08);
                        console::put(b' ');
                        console::put(0x08);
                    }

                    line.pop();
                }
                continue;
            } else if c == b'\r' || c == b'\n' {
                // newline
                if line.is_empty() {
                    console::print("\r\n> ");
                    continue;
                }

                if let Ok(line_u8) = alloc::str::from_utf8(&line) {
                    if !IS_STD {
                        console::print("\r\n");
                    }

                    // Evaluate the line.
                    eval(line_u8, &blisp_ctx);

                    console::print("\r\n> ");
                } else {
                    console::print("Error: Input string is not UTF-8.");
                }

                line.clear();
            } else {
                // normal character

                if !IS_STD {
                    console::put(c); // echo back
                }

                line.push(c);
            }
        }

        sleep(Duration::from_millis(20)).await;
    }
}

fn eval(expr: &str, ctx: &blisp::semantics::Context) {
    match blisp::eval(expr, ctx) {
        Ok(results) => {
            for result in results {
                match result {
                    Ok(msg) => {
                        console::print(&msg);
                    }
                    Err(e) => {
                        console::print(&e);
                        console::print("\r\n\r\ntry as follows\r\n> (help)\r\n");
                    }
                }
            }
        }
        Err(err) => {
            console::print(&err.msg);
            console::print("\r\n\r\ntry as follows\r\n> (help)\r\n");
        }
    }
}

const CODE: &str = "(export factorial (n) (Pure (-> (Int) Int))
    (factorial' n 1))

(defun factorial' (n total) (Pure (-> (Int Int) Int))
    (if (<= n 0)
        total
        (factorial' (- n 1) (* n total))))

(export help () (IO (-> () []))
    (help_ffi))

(export task () (IO (-> () []))
    (task_ffi))

(export interrupt () (IO (-> () []))
    (interrupt_ffi))

(export ifconfig () (IO (-> () []))
    (ifconfig_ffi))

(export netdump (interface_id) (IO (-> (Int) []))
    (netdump_ffi interface_id))

(export add_ipv4 (interface_id a b c d prefix_len) (IO (-> (Int Int Int Int Int Int) []))
    (add_ipv4_ffi interface_id a b c d prefix_len))

(export arping4 (interface_id a b c d) (IO (-> (Int Int Int Int Int) []))
    (arping4_ffi interface_id a b c d))

(export set_gateway4 (interface_id a b c d) (IO (-> (Int Int Int Int Int) []))
    (set_gateway4_ffi interface_id a b c d))

(export reboot () (IO (-> () []))
    (reboot_ffi))

(export shutdown () (IO (-> () []))
    (shutdown_ffi))
";

const PERF_CODE: &str = "(export perf () (IO (-> () []))
    (perf_ffi))";

#[embedded]
fn help_ffi() {
    console::print("Awkernel v202306\r\n");
    console::print("BLisp grammer: https://ytakano.github.io/blisp/\r\n\r\n");

    console::print("BLisp functions:\r\n");

    let mut lines = String::new();

    lines.push_str("(help)      ; print this message\r\n");
    lines.push_str("(task)      ; print tasks\r\n");
    lines.push_str("(interrupt) ; print interrupt information\r\n");
    lines.push_str("(ifconfig)  ; print network interfaces\r\n");
    lines.push_str("(netdump if); dump device registers\r\n");
    lines.push_str("(add_ipv4 if a b c d prefix) ; add IPv4 address\r\n");
    lines.push_str("(arping4 if a b c d)         ; send ARP request\r\n");
    lines.push_str("(set_gateway4 if a b c d)    ; set IPv4 gateway\r\n");
    lines.push_str("(reboot)    ; reboot x86_64 systems\r\n");
    lines.push_str("(shutdown)  ; power off x86_64 systems\r\n");

    #[cfg(feature = "perf")]
    lines.push_str("(perf)      ; print performance information\r\n");

    console::print(lines.as_str());
}

#[embedded]
fn task_ffi() {
    let msg = format!("Uptime: {}\r\n", awkernel_async_lib::uptime(),);
    console::print(&msg);

    print_tasks();

    console::print("\r\n");

    let msg = format!(
        "Total preemption: {}\r\n",
        awkernel_async_lib::task::get_num_preemption(),
    );
    console::print(&msg);

    console::print("Running Tasks:\r\n");
    for task in awkernel_async_lib::task::get_tasks_running().iter() {
        let msg = if task.task_id != 0 {
            format!("  cpu: {:>3}, task: {:>5}\r\n", task.cpu_id, task.task_id)
        } else {
            format!("  cpu: {:>3}, task:\r\n", task.cpu_id)
        };
        console::print(&msg);
    }
}

#[embedded]
fn interrupt_ffi() {
    let handlers = awkernel_lib::interrupt::get_handlers();

    console::print("IRQ Name\r\n");
    for (k, v) in handlers.iter() {
        let msg = format!("{k:>3} name: {v}\r\n");
        console::print(&msg);
    }
}

#[embedded]
fn ifconfig_ffi() {
    let ifs = awkernel_lib::net::get_all_interface();
    for netif in ifs.iter() {
        let msg = format!("{netif}\r\n\r\n");
        console::print(&msg);
    }
}

#[embedded]
fn netdump_ffi(interface_id: BigInt) {
    let Some(interface_id) = to_u64_arg("interface_id", &interface_id) else {
        return;
    };

    if let Err(e) = awkernel_lib::net::debug_dump_interface(interface_id) {
        console::print(&format!("netdump failed: {e}\r\n"));
    }
}

fn to_u64_arg(name: &str, value: &BigInt) -> Option<u64> {
    let Some(value) = value.to_u64() else {
        console::print(&format!(
            "{name} must be a non-negative integer that fits in u64\r\n"
        ));
        return None;
    };

    Some(value)
}

fn to_u8_arg(name: &str, value: &BigInt) -> Option<u8> {
    let Some(value) = value.to_u8() else {
        console::print(&format!("{name} must be an integer in 0..=255\r\n"));
        return None;
    };

    Some(value)
}

#[embedded]
fn add_ipv4_ffi(
    interface_id: BigInt,
    a: BigInt,
    b: BigInt,
    c: BigInt,
    d: BigInt,
    prefix_len: BigInt,
) {
    let Some(interface_id) = to_u64_arg("interface_id", &interface_id) else {
        return;
    };
    let Some(a) = to_u8_arg("IPv4 octet", &a) else {
        return;
    };
    let Some(b) = to_u8_arg("IPv4 octet", &b) else {
        return;
    };
    let Some(c) = to_u8_arg("IPv4 octet", &c) else {
        return;
    };
    let Some(d) = to_u8_arg("IPv4 octet", &d) else {
        return;
    };
    let Some(prefix_len) = to_u8_arg("prefix_len", &prefix_len) else {
        return;
    };

    awkernel_lib::net::add_ipv4_addr(
        interface_id,
        core::net::Ipv4Addr::new(a, b, c, d),
        prefix_len,
    );
}

#[embedded]
fn set_gateway4_ffi(interface_id: BigInt, a: BigInt, b: BigInt, c: BigInt, d: BigInt) {
    let Some(interface_id) = to_u64_arg("interface_id", &interface_id) else {
        return;
    };
    let Some(a) = to_u8_arg("IPv4 octet", &a) else {
        return;
    };
    let Some(b) = to_u8_arg("IPv4 octet", &b) else {
        return;
    };
    let Some(c) = to_u8_arg("IPv4 octet", &c) else {
        return;
    };
    let Some(d) = to_u8_arg("IPv4 octet", &d) else {
        return;
    };

    if let Err(e) = awkernel_lib::net::set_default_gateway_ipv4(
        interface_id,
        core::net::Ipv4Addr::new(a, b, c, d),
    ) {
        console::print(&format!("set_gateway4 failed: {e}\r\n"));
    }
}

#[embedded]
fn arping4_ffi(interface_id: BigInt, a: BigInt, b: BigInt, c: BigInt, d: BigInt) {
    let Some(interface_id) = to_u64_arg("interface_id", &interface_id) else {
        return;
    };
    let Some(a) = to_u8_arg("IPv4 octet", &a) else {
        return;
    };
    let Some(b) = to_u8_arg("IPv4 octet", &b) else {
        return;
    };
    let Some(c) = to_u8_arg("IPv4 octet", &c) else {
        return;
    };
    let Some(d) = to_u8_arg("IPv4 octet", &d) else {
        return;
    };

    let Ok(interface) = awkernel_lib::net::get_interface(interface_id) else {
        console::print("arping4 failed: invalid interface\r\n");
        return;
    };

    let Some((src_ip, _)) = interface.ipv4_addrs.first().copied() else {
        console::print("arping4 failed: interface has no IPv4 address\r\n");
        return;
    };

    let src_mac = interface.mac_address;
    let target_ip = core::net::Ipv4Addr::new(a, b, c, d);

    let mut frame = Vec::with_capacity(60);
    frame.extend_from_slice(&[0xff; 6]);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&awkernel_lib::net::ethertypes::ETHER_TYPE_ARP.to_be_bytes());
    frame.extend_from_slice(&1u16.to_be_bytes());
    frame.extend_from_slice(&awkernel_lib::net::ethertypes::ETHER_TYPE_IP.to_be_bytes());
    frame.push(6);
    frame.push(4);
    frame.extend_from_slice(&1u16.to_be_bytes());
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&src_ip.octets());
    frame.extend_from_slice(&[0; 6]);
    frame.extend_from_slice(&target_ip.octets());
    frame.resize(60, 0);

    for _ in 0..64 {
        if let Err(e) = awkernel_lib::net::raw_send(interface_id, 0, &frame) {
            console::print(&format!("arping4 failed: {e}\r\n"));
            break;
        }
    }
}

#[embedded]
fn reboot_ffi() {
    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    {
        awkernel_lib::arch::x86_64::power::reboot();
    }

    #[cfg(not(all(target_arch = "x86_64", target_os = "none")))]
    {
        console::print("reboot is unsupported on this architecture\r\n");
    }
}

#[embedded]
fn shutdown_ffi() {
    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    {
        awkernel_lib::arch::x86_64::power::shutdown();
    }

    #[cfg(not(all(target_arch = "x86_64", target_os = "none")))]
    {
        console::print("shutdown is unsupported on this architecture\r\n");
    }
}

#[cfg(feature = "perf")]
#[embedded]
fn perf_ffi() {
    console::print("Perform non-primary CPU [tsc]:\r\n");
    console::print(" cpu | Type           |   kernel_time  |    task_time   |    idle_time   | interrupt_time | context_switch |    perf_time   \r\n");
    console::print("=====|================|================|================|================|================|================|================\r\n");

    use awkernel_async_lib::task::perf;

    for cpu_id in 0..awkernel_lib::cpu::num_cpu() {
        let kernel_time = perf::get_kernel_time(cpu_id);
        let task_time = perf::get_task_time(cpu_id);
        let idle_time = perf::get_idle_time(cpu_id);
        let interrupt_time = perf::get_interrupt_time(cpu_id);
        let contxt_switch_time = perf::get_context_switch_time(cpu_id);
        let perf_time = perf::get_perf_time(cpu_id);

        let msg = format!(
            "{cpu_id:>4} | Total          |{kernel_time:>15} |{task_time:>15} |{idle_time:>15} |{interrupt_time:>15} |{contxt_switch_time:>15} |{perf_time:>15}\r\n"
        );

        console::print(&msg);

        let ave_kernel_time = perf::get_ave_kernel_time(cpu_id).unwrap_or(0.0);
        let ave_task_time = perf::get_ave_task_time(cpu_id).unwrap_or(0.0);
        let ave_idle_time = perf::get_ave_idle_time(cpu_id).unwrap_or(0.0);
        let ave_interrupt_time = perf::get_ave_interrupt_time(cpu_id).unwrap_or(0.0);
        let ave_contxt_switch_time = perf::get_ave_context_switch_time(cpu_id).unwrap_or(0.0);
        let ave_perf_time = perf::get_ave_perf_time(cpu_id).unwrap_or(0.0);

        let msg_ave = format!(
            "     | Avg            | {ave_kernel_time:>14.2} | {ave_task_time:>14.2} |{ave_idle_time:>15.2} |{ave_interrupt_time:>15.2} |{ave_contxt_switch_time:>15.2} |{ave_perf_time:>15.2}\r\n",
        );

        console::print(&msg_ave);

        let worst_kernel_time = perf::get_kernel_wcet(cpu_id);
        let worst_task_time = perf::get_task_wcet(cpu_id);
        let worst_idle_time = perf::get_idle_wcet(cpu_id);
        let worst_interrupt_time = perf::get_interrupt_wcet(cpu_id);
        let worst_contxt_switch_time = perf::get_context_switch_wcet(cpu_id);
        let worst_perf_time = perf::get_perf_wcet(cpu_id);

        let msg_worst = format!(
            "     | Worst          | {worst_kernel_time:>14} | {worst_task_time:>14} |{worst_idle_time:>15} |{worst_interrupt_time:>15} |{worst_contxt_switch_time:>15} |{worst_perf_time:>15}\r\n",
        );

        console::print(&msg_worst);

        if cpu_id < awkernel_lib::cpu::num_cpu() - 1 {
            console::print("-----|----------------|----------------|----------------|----------------|----------------|----------------|----------------\r\n");
        }
    }
}

fn print_tasks() {
    let tasks = task::get_tasks();

    console::print("Tasks:\r\n");

    let msg = format!(
        "{:>5}  {:<10} {:>14} {:>14} name\r\n",
        "ID", "State", "#Preemption", "Last Executed"
    );
    console::print(&msg);

    for t in tasks {
        let mut node = MCSNode::new();
        let info = t.info.lock(&mut node);

        let state = format!("{:?}", info.get_state());

        let msg = format!(
            "{:>5}{} {:<10} {:>14} {:>14} {}\r\n",
            t.id,
            if info.panicked() { "*" } else { " " },
            state,
            info.get_num_preemption(),
            info.get_last_executed().uptime().as_micros(),
            t.name,
        );

        console::print(&msg);
    }
}
