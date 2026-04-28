use core::net::Ipv4Addr;

use super::IpAddr;
#[cfg(feature = "baseline_trace")]
use crate::{
    baseline_trace::{UnblockKind, WaitClass},
    task,
};
use awkernel_lib::net::{udp_socket::SockUdp, NetManagerError};
use futures::Future;
use pin_project::pin_project;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UdpSocketError {
    SocketCreationError,
    SendError,
    InterfaceIsNotReady,
    MulticastInvalidIPv4Address,
    MulitcastError,
}

#[derive(Debug, Clone)]
pub struct UdpConfig {
    pub addr: IpAddr,
    pub port: Option<u16>,
    pub rx_buffer_size: usize,
    pub tx_buffer_size: usize,
}

impl Default for UdpConfig {
    fn default() -> Self {
        UdpConfig {
            addr: IpAddr::new_v4(Ipv4Addr::new(0, 0, 0, 0)),
            port: None,
            rx_buffer_size: 1024 * 64,
            tx_buffer_size: 1024 * 64,
        }
    }
}

pub struct UdpSocket {
    socket_handle: awkernel_lib::net::udp_socket::UdpSocket,
}

impl UdpSocket {
    /// Create a new UDP socket.
    ///
    /// On std environments `interface_id`, `config.rx_buffer_size`, and `config.tx_buffer_size` are ignored.
    ///
    /// # Example
    ///
    /// ```
    /// use awkernel_async_lib::net::{IpAddr, udp::{UdpConfig, UdpSocket}};
    ///
    /// const INTERFACE_ID: u64 = 0;
    ///
    /// async fn example_udp_socket() {
    ///     let mut socket = UdpSocket::bind_on_interface(
    ///         INTERFACE_ID,
    ///         &UdpConfig {
    ///             addr: IpAddr::new_v4(core::net::Ipv4Addr::new(192, 168, 0, 1)),
    ///             port: Some(10000),
    ///             ..Default::default()
    ///         },
    ///     )
    ///     .unwrap();
    /// }
    /// ```
    pub fn bind_on_interface(
        interface_id: u64,
        config: &UdpConfig,
    ) -> Result<UdpSocket, UdpSocketError> {
        let socket_handle = awkernel_lib::net::udp_socket::UdpSocket::bind_on_interface(
            interface_id,
            &config.addr,
            config.port,
            config.rx_buffer_size,
            config.tx_buffer_size,
        )
        .or(Err(UdpSocketError::SocketCreationError))?;

        Ok(UdpSocket { socket_handle })
    }

    /// Send a UDP packet to the specified address and port.
    #[inline(always)]
    pub async fn send(
        &mut self,
        data: &[u8],
        dst_addr: &IpAddr,
        dst_port: u16,
    ) -> Result<(), UdpSocketError> {
        UdpSender {
            socket: self,
            data,
            dst_addr,
            dst_port,
            blocked_task_id: None,
        }
        .await
    }

    /// Receive a UDP packet from the socket.
    /// This function returns the number of bytes read, the source address, and the source port.
    ///
    /// If the length of the received data is greater than the length of the buffer,
    /// the data is truncated to the length of the buffer.
    #[inline(always)]
    pub async fn recv(&mut self, buf: &mut [u8]) -> Result<(usize, IpAddr, u16), UdpSocketError> {
        UdpReceiver {
            socket: self,
            buf,
            blocked_task_id: None,
        }
        .await
    }

    /// Join a multicast group.
    #[inline(always)]
    pub fn join_multicast_v4(
        &mut self,
        multicast_addr: Ipv4Addr,
        interface_addr: Ipv4Addr,
    ) -> Result<(), UdpSocketError> {
        match self
            .socket_handle
            .join_multicast_v4(multicast_addr, interface_addr)
        {
            Ok(()) => Ok(()),
            Err(NetManagerError::SendError) => Err(UdpSocketError::SendError),
            Err(NetManagerError::MulticastInvalidIpv4Address) => {
                Err(UdpSocketError::MulticastInvalidIPv4Address)
            }
            Err(e) => {
                log::debug!("{e:?}");
                Err(UdpSocketError::MulitcastError)
            }
        }
    }

    #[inline(always)]
    pub fn leave_multicast_v4(
        &mut self,
        multicast_addr: Ipv4Addr,
        interface_addr: Ipv4Addr,
    ) -> Result<(), UdpSocketError> {
        match self
            .socket_handle
            .leave_multicast_v4(multicast_addr, interface_addr)
        {
            Ok(()) => Ok(()),
            Err(NetManagerError::SendError) => Err(UdpSocketError::SendError),
            Err(NetManagerError::MulticastInvalidIpv4Address) => {
                Err(UdpSocketError::MulticastInvalidIPv4Address)
            }
            Err(_) => Err(UdpSocketError::MulitcastError),
        }
    }
}

#[pin_project]
pub struct UdpSender<'a> {
    socket: &'a mut UdpSocket,
    data: &'a [u8],
    dst_addr: &'a IpAddr,
    dst_port: u16,
    blocked_task_id: Option<u32>,
}

impl Future for UdpSender<'_> {
    type Output = Result<(), UdpSocketError>;
    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        let this = self.project();

        match this.socket.socket_handle.send_to(
            this.data,
            this.dst_addr,
            *this.dst_port,
            cx.waker(),
        ) {
            Ok(true) => {
                record_io_ready(this.blocked_task_id);
                core::task::Poll::Ready(Ok(()))
            }
            Ok(false) => {
                record_io_block(this.blocked_task_id);
                core::task::Poll::Pending
            }
            Err(NetManagerError::InterfaceIsNotReady) => {
                record_io_ready(this.blocked_task_id);
                core::task::Poll::Ready(Err(UdpSocketError::InterfaceIsNotReady))
            }
            Err(_) => {
                record_io_ready(this.blocked_task_id);
                core::task::Poll::Ready(Err(UdpSocketError::SendError))
            }
        }
    }
}

#[pin_project]
pub struct UdpReceiver<'a> {
    socket: &'a mut UdpSocket,
    buf: &'a mut [u8],
    blocked_task_id: Option<u32>,
}

impl Future for UdpReceiver<'_> {
    type Output = Result<(usize, IpAddr, u16), UdpSocketError>;
    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        let this = self.project();

        let (socket, buf) = (this.socket, this.buf);

        match socket.socket_handle.recv(buf, cx.waker()) {
            Ok(Some(result)) => {
                record_io_ready(this.blocked_task_id);
                core::task::Poll::Ready(Ok(result))
            }
            Ok(None) => {
                record_io_block(this.blocked_task_id);
                core::task::Poll::Pending
            }
            Err(_) => {
                record_io_ready(this.blocked_task_id);
                core::task::Poll::Ready(Err(UdpSocketError::SendError))
            }
        }
    }
}

#[cfg(feature = "baseline_trace")]
fn record_io_block(blocked_task_id: &mut Option<u32>) {
    if blocked_task_id.is_none() {
        *blocked_task_id = task::record_current_task_block(WaitClass::Io);
    }
}

#[cfg(not(feature = "baseline_trace"))]
fn record_io_block(_blocked_task_id: &mut Option<u32>) {}

#[cfg(feature = "baseline_trace")]
fn record_io_ready(blocked_task_id: &mut Option<u32>) {
    if let Some(task_id) = blocked_task_id.take() {
        task::record_task_unblock(task_id, WaitClass::Io, UnblockKind::Ready);
    }
}

#[cfg(not(feature = "baseline_trace"))]
fn record_io_ready(_blocked_task_id: &mut Option<u32>) {}
