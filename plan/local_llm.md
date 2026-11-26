# Awkernel 上でのローカル LLM 推論 — 計画 (v4)

状態: **デモ完成 (段階 0〜7 完了、2026-11-26)**。結果は 5.1〜5.4 節、決定事項は 7 章、残課題は 8 章。手順書は `applications/local_llm/README.md`。

## 1. 目標

[llm.asm](https://github.com/alesha-pro/llm.asm) の Phase 2 (bare-metal Qwen3-0.6B OS) と同様に、
Awkernel (x86_64) のカーネル空間で LLM (Qwen3-0.6B, Q8_0 量子化) を CPU のみで推論し、
シリアルコンソールから対話できるようにする。動作確認は Linux KVM (QEMU `-enable-kvm -cpu host`) で行う。

非目標 (当面): GPU、学習 (llm.asm Phase 3)、AArch64/RISC-V での高速化、Qwen3-0.6B 以外のモデル。

## 2. llm.asm の調査結果

| 項目 | 内容 |
|---|---|
| モデル | Qwen/Qwen3-0.6B。28 層, dim=1024, hidden=3072, 16 Q ヘッド / 8 KV ヘッド (GQA), head_dim=128, vocab=151,936, 埋め込みと LM head は共有 |
| 構成要素 | RMSNorm, QK-Norm, RoPE (theta=1e6), SwiGLU, KV キャッシュ, greedy / temperature + top-p サンプリング |
| 重み形式 | 独自 `.asmbin` (v2: 614 MiB)。128 B ヘッダ + トークナイザ節 + 行単位 Q8_0 (int8 × K + ブロック 32 要素ごとの FP16 スケール)。全節 64 B アライン、ゼロコピー前提 |
| トークナイザ | BPE。マージ表は export 時に構築済みのオープンアドレス・ハッシュ表 (524,288 スロット) として `.asmbin` に同梱。Qwen3 チャットテンプレートはコード側 |
| 演算核 | AVX2 + FMA + F16C 必須。`FAST_Q8=1` は活性化も int8 化し `vpmaddubsw`/`vpmaddwd` で INT8×INT8 GEMV |
| 並列化 | 行列の行範囲を全コアに分割するスピンバリア (bare-metal) / futex スレッドプール (Linux) |
| bare-metal 部 | Multiboot1 起動、モデルは Multiboot モジュール (QEMU `-initrd`) として物理メモリに置かれゼロコピー参照。2 MiB ページの恒等写像。KV キャッシュ + 作業領域 449 MiB (ctx=2048) |
| 性能 (作者計測, EPYC 7642 KVM) | 1 vCPU 23.7 tok/s、16 vCPU 134.9 tok/s |
| 共有コアの外部依存 | `sys_mmap_anon`, `sys_munmap`, `tp_matmul_q8_0`, `tp_gqa_attn`, `sys_write_all`, `str_len` のみ (`-DBARE_METAL` 時) |
| ライセンス | MIT (モデルは Apache-2.0) |

## 3. Awkernel 側の現状 (調査結果)

- **FPU/SIMD**: `enable_fpu()` (`kernel/src/arch/x86_64/kernel_main.rs`) は SSE のみ有効化 (CR4.OSFXSR)。
  **CR4.OSXSAVE / XCR0 は未設定なので現状 AVX 命令は #UD になる**。BSP・AP の両方で呼ばれている。
- **コンテキストスイッチ**: `awkernel_lib/src/context/x86_64.rs` は callee-saved の汎用レジスタのみ保存。
  自発的な切替は SysV ABI 上 XMM/YMM が caller-saved なので問題ない。
  一方、プリエンプション (`extern "x86-interrupt" fn preemption`) では、YMM 上位 128 bit が保存されない
  (XMM はハンドラのプロローグで保存されるはず — 要確認)。AVX を使うタスク同士が同一 CPU 上で
  プリエンプトし合うと状態が壊れる可能性がある。
- **ターゲット**: `targets/x86_64-kernel.json` に soft-float 指定は無く、SSE2 はカーネルで使用可。
  AVX2 は関数単位の `#[target_feature(enable = "avx2,fma,f16c")]` で使える (ターゲット全体は変えない)。
- **メモリ**: 起動時に Usable な全物理フレームを 4 KiB ページで `HEAP_START` 以降に写像し primary heap とする。
  最大ブロックは約 8 GiB なので数百 MiB の確保は可能。2 MiB ページ写像は無い。
- **ブート**: `bootloader` 0.11 (UEFI)。`UefiBoot::set_ramdisk()` と `BootInfo::ramdisk_addr/ramdisk_len` があり、
  Multiboot モジュール相当の仕組みが使える (`x86bootdisk` は未対応)。
- **ストレージ**: NVMe ドライバはあるが I/O 経路 (`_submit_io`) は `StorageDevice` に未接続。virtio-blk はスタブ。
  FAT はメモリ上ディスクのみ。→ ディスクからのモデル読み込みは追加実装が大きい。
- **実行モデル**: CPU 0 は IRQ/タイマ専用でタスクを実行しない。ワーカは CPU 1..n。
  スケジューラは EDF / LLFT (CpuSet 指定可) と PrioritizedFIFO / RR。
- **コンソール**: `awkernel_shell` (BLisp REPL) が `console::get()` を単独で消費している。
- **QEMU**: 既存 `make qemu-x86_64` は KVM 無効・`-cpu` 既定 (AVX2 無し)・4 GiB を 1 GiB × 4 NUMA ノードに分割。
- **ホスト**: `/dev/kvm` あり、AVX2/FMA/F16C あり、16 スレッド、62 GiB RAM、QEMU 8.2.2。`nasm` は未インストール。

## 4. 方針 (提案)

### 4.1 推論エンジン: Rust で再実装する (推奨)

クレートを 2 つに分ける (決定)。`.asmbin` 形式と `export_weights.py` はそのまま流用する。

- **エンジンクレート** (仮称 `awkernel_llm`、ワークスペース直下): no_std + alloc のみに依存し、**Awkernel のクレートに依存しない**。
  `.asmbin` パーサ、トークナイザ、forward、サンプリング、SIMD 核 (`core::arch::x86_64` intrinsics) とスカラ核。
  入力は `&[u8]` (モデル)、作業メモリ、Executor (4.4)。`cargo test` でホスト単体テストできる。
- **アプリクレート** (`applications/local_llm`): ramdisk の取得、LLM タスクの生成、シェルとの接続、コンソール出力。

理由:
- Awkernel は `std` フィーチャで Linux プロセスとしても動くため、**同じコードをまずホストで開発・単体テストできる**。
- llm.asm (Phase 1 の Linux バイナリ) を正解器として、トークン列・logits を突き合わせられる。
- スカラ実装を併設すれば AArch64 等にも将来持っていける。ビルドに `nasm` を要求しない。
- Awkernel の「Rust で安全に」という方針と整合する。unsafe は SIMD 核とゼロコピー参照に局所化する。

代替案 (B): llm.asm の `shared/*.asm` を `nasm` でアセンブルしてリンクし、上記 6 シンボルだけ Rust で実装する。
最短で動くが、ビルド依存 (`nasm`) が増え、x86_64 専用、ホストテスト不可、タスク/プリエンプションとの協調が難しい。
→ 「まず動くものを最速で見たい」場合の選択肢として残す。

### 4.2 モデルの渡し方: ブートローダの ramdisk (推奨)

`x86bootdisk` に `--ramdisk <path>` を追加し、`.asmbin` を UEFI イメージに同梱。
カーネルは `boot_info.ramdisk_addr/len` を `awkernel_lib` 経由でアプリに公開し、ゼロコピーで参照する。
llm.asm の Multiboot モジュール方式に最も近く、実機でも同じ手順で動く。

懸念と代替:
- 614 MiB の ramdisk を UEFI ブートローダが読むのに時間がかかる可能性、物理的に連続な領域が必要
  (現在の 1 GiB × 4 NUMA 構成では取れない恐れ → LLM 用 QEMU ターゲットではメモリを 8 GiB 程度にする)。
- 代替 1: QEMU `fw_cfg` (`-fw_cfg name=opt/awkernel/model,file=...`) + 小さな fw_cfg DMA ドライバ。
  イメージ再生成が不要で速いが QEMU/KVM 専用。
- 代替 2: NVMe の I/O 経路を完成させて raw/FAT から読む。実機向けの本筋だが作業量が大きい。
- 代替 3: ネットワーク (TCP) で受信。既存スタックで可能だが起動のたびに転送が必要。

### 4.3 AVX の有効化とプリエンプション

1. BSP/AP 共通で CPUID 確認の上 CR4.OSXSAVE を立て XCR0=7 (x87|SSE|AVX) を設定する。
   AVX2/FMA/F16C が無い CPU ではスカラ実装にフォールバック (もしくはアプリ起動を拒否)。
2. プリエンプション時の YMM 保護 (詳細は 4.3.1)。**推奨は (a') プリエンプション割り込みハンドラ内でのスタックローカル XSAVE** (未決定)。

#### 4.3.1 プリエンプションと YMM 状態の詳細

**どこで状態が失われるか**

x86_64 でタスクが CPU を手放す経路は 3 つある。

| 経路 | 仕組み | YMM の扱い |
|---|---|---|
| `.await` による協調的な切替 | Future が `Pending` を返し、ワーカスレッドのループに戻る | 通常の関数リターン。問題なし |
| 自発的プリエンプション (`voluntary_preemption`) | `InterruptGuard` の drop やロック解放から通常の関数呼び出しで `preemption()` → `context_switch` | SysV ABI では XMM/YMM は全て caller-saved。呼び出し側のコンパイラが退避済みなので問題なし |
| 割り込みによるプリエンプション (IRQ 255) | `extern "x86-interrupt" fn preemption` がタスクのスタック上で動き、`handle_preemption()` → `do_preemption()` → `context_switch` | **XMM0–15 はハンドラのプロローグがスタックに保存するが、YMM 上位 128 bit は保存されない** |

3 つ目は実測で確認した: カーネルと同じ SSE のみのビルドでは `x86-interrupt` 関数は `movaps %xmmN, (%rsp)` を 16 本出すが YMM は保存しない
(`-C target-feature=+avx` を付けた場合のみ `vmovups %ymmN` になる)。

なお、IRQ 255 以外の割り込みハンドラ (`irq_handler!`) は `handle_irq()` を呼ぶだけで切替を起こさない。
ハンドラ内は IF=0 なので `InterruptGuard::drop` 経由の自発的プリエンプションも発火しない
(`awkernel_sync` 側のロック解放も同じ条件か要確認)。カーネル本体は VEX 無しの SSE 命令でコンパイルされており、
レガシー SSE 命令は YMM 上位を変更しないので、割り込みハンドラ自体が YMM 上位を壊すこともない。
つまり**問題は IRQ 255 のハンドラ 1 か所に局所化されている**。

**壊れる具体的なシナリオ**

LLM タスク A が AVX2 の内積ループの途中 (ymm レジスタに累積値がある状態) で IRQ 255 を受ける。

1. より優先度の高いタスク B に切り替わる。B が AVX を使えば (別の LLM ワーカなど)、A の YMM 上位が上書きされる。
   B の AVX 関数が末尾で `vzeroupper` を実行するだけでも A の上位はゼロになる。
2. B が AVX を使わなくても、A は**別の CPU で再開されうる** (`yield_preempted_and_wake_task` のコメント参照)。
   その CPU の YMM 上位は無関係な残骸なので、やはり壊れる。

結果は例外ではなく**サイレントな計算誤り** (トークン列が変わる) になる。
LLM タスクを最低優先度で動かすデモ構成でも、シェル・ロガー・ネットワークサービスなど優先度の高いタスクが
起床するたびにプリエンプションが起きるので、「LLM タスクが 1 つだけなら安全」とは言えない。

**対策案**

- **(a') IRQ 255 ハンドラ内でスタックローカルに XSAVE/XRSTOR する (推奨)**
  ```
  extern "x86-interrupt" fn preemption(..) {
      ...
      let mut area = XSaveArea::new();          // 64 B アライン、スタック上 (x87+SSE+AVX で 832 B)
      if avx_enabled { xsave(&mut area, AVX) }  // 要求ビットマスク = AVX (YMM 上位) のみ
      handle_preemption();                      // ここで他タスクへ切替。再開時 (別 CPU でも可) にここへ戻る
      if avx_enabled { xrstor(&area, AVX) }
  }
  ```
  - 退避先がプリエンプトされたタスク自身のスタックなので、CPU 間の移動があっても正しい。タスクごとのフラグや領域管理が不要。
  - 変更は `kernel/src/arch/x86_64/interrupt_handler.rs` の 1 関数と、XSAVE の薄いラッパのみ。
    **検証済みの `context_switch` (`awkernel_lib/src/context/x86_64.rs`) と `Context` 構造体には手を入れない**。
  - コストはプリエンプション発生時のみ。256 B の退避/復元で、切替処理全体 (ロック取得・キュー操作) に比べ十分小さい。
    Rust が生成する AVX 関数は終了時に `vzeroupper` するので、AVX 非使用タスクでは YMM 上位は初期状態であり、
    XSAVEOPT が使えれば退避自体がほぼ省略される。
  - XMM は従来どおりプロローグが保存する。MXCSR/x87 制御語は現状どおり保存しない (タスクが変更しない前提。現状と同じ)。
- (a) タスク構造体に XSAVE 領域を持たせ `context_switch` で保存する — 一般的な OS の方式だが、
  検証済みコンポーネントの変更になり、自発的切替 (保存不要) にもコストが乗る。(a') で足りるので採らない。
- (b) SIMD 区間だけ割り込みを禁止する (Linux の `kernel_fpu_begin/end` 相当) — 実装は最小だが、
  LLM ワーカの CPU で数百 µs 単位の割り込み禁止区間が毎トークン多数発生する。デモは動くが、
  後で狙うリアルタイム性評価と相容れない。(a') が動くまでの暫定手段としてのみ。
- (c) 構成で回避 (LLM を最高優先度にしてプリエンプトされないようにする等) — PrioritizedFIFO に CPU 固定は無く、
  保証が弱い。シェル等を飢餓させる。採らない。
- (d) カーネル全体を `+avx` でビルドし全割り込みハンドラに YMM を保存させる — 全割り込みで 512 B 余分に退避、
  AVX の無い CPU (QEMU TCG 既定) で起動不能、XCR0 設定前に AVX 命令が走る危険。採らない。

**検証方法**

`applications/tests/test_avx_preempt` (仮) を追加する: 低優先度タスクが `asm!` で ymm0–15 に既知パターンを載せて
スピンしながら照合し、高優先度の周期タスクが AVX で YMM を上書きして頻繁にプリエンプトする。
対策前に失敗し、対策後に通ることを KVM 上で確認する (段階 3)。

**将来の注意**: AVX-512 を有効化する場合は XCR0 と XSAVE のマスクに opmask/ZMM を追加する (領域は約 2.7 KiB に増える)。

### 4.4 タスク構成と並列化

**「シングルタスク」の意味を 2 つに分ける**

- (S0) シェルタスクの中で同期的に推論する (`(llm "...")` の FFI 内で全トークン生成)。**採らない**。
  - BLisp の FFI は同期関数で、シェル・ロガー・ネットワークサービスは全て `PrioritizedFIFO(0)`。
    同優先度間はプリエンプトされないので、応答生成の数秒〜数十秒間ワーカ CPU を 1 つ占有し続ける。
    ワーカが 1 つ (`-smp 2`) だと他のサービスが全て止まる。
  - コンソール入力を読むのがシェルタスク自身なので、生成の中断 (Ctrl-C) ができない。
  - 推論の優先度がシェルの優先度に固定され、後から「LLM は低優先度のベストエフォート」にできない。
- (S1) 専用の LLM タスク 1 つで推論する。シェルの `(llm "...")` はチャネルで要求を渡すだけ。
  LLM タスクはトークンごとに yield し、出力をコンソールへストリームする。**デモ v1 はこれ**。

**S1 に残る制約**: 速度のみ。llm.asm の実測で 1 コア約 20 tok/s (プロンプト処理も同速度なので、
30 トークン程度のチャットテンプレートで最初の 1 文字まで 1.5 秒前後)。Qwen3 の思考モードは数百トークン出すので遅く感じる。
4 KiB ページのため llm.asm より 1 割程度遅い可能性がある。

**マルチタスク (行分割並列) の利点**

- 速度: llm.asm の実測で 4 コア 3.4 倍、8 コア 5 倍、16 コアで頭打ち (メモリ帯域律速。重み 614 MiB を毎トークン全走査するため)。
  開発機 (Ryzen 9 5900HX, 8C/16T, L3 16 MiB) では 4〜6 ワーカ付近で飽和する見込み。
- Awkernel のデモとしての価値: async タスクとマルチコアスケジューラの上で LLM が動くことを示せる。
  LLM ワーカ同士・他タスクとのプリエンプションで 4.3.1 の XSAVE 経路が実際に使われる。
- 後のリアルタイム共存評価の土台になる (LLM ワーカ = 低優先度タスク群、RT タスクがそれをプリエンプト)。
- 行ごとに独立な計算なので、分割数によらず出力はビット同一 (llm.asm でも確認済みの性質)。

**マルチタスクのコストとリスク**

- 並列区間が 1 トークンあたり約 140 回 (28 層 × QKV・attention・o_proj・gate+up・down) + LM head。
  100 tok/s を狙うと 1 区間の予算は約 70 µs。タスク起床 (スケジューラのロック、スリープ中 CPU への IPI) の遅延が
  数十 µs あると効果が消える → ワーカはスピン待ちが必要になり、待機中も CPU を消費する (RT 評価時に再設計が要る)。
- バリア中にワーカの 1 つがプリエンプトされると、全員がそのワーカを待つ (KVM ではホストによる vCPU 横取りでも同じ)。
- CPU 0 はタスクを実行しないので、使えるのは `smp - 1` コア。
- 正しさの切り分けが難しくなる。まず単一タスクでトークン一致を取ってから並列化する。

**方針**

- エンジンの演算核は最初から「行範囲 `[r0, r1)` を処理する関数」として書き、エンジンは
  「範囲を分割して実行する」抽象 (`trait Executor { fn par_for(&self, n: usize, f: &(dyn Fn(Range<usize>) + Sync)); }` 相当) にのみ依存する。
  デモ v1 はインライン実行の Executor、ホストテストは `std::thread`、Awkernel 版はワーカタスク群を後から差し込む。
  これによりシングル → マルチは Executor の追加だけで済み、エンジンの書き直しは発生しない。
- 順序は「S1 で動作・トークン一致を確認 → 対話 UI → 並列化」。**並列化はデモの範囲に含める** (決定)。
  性能チューニング (2 MiB ページ、起床遅延の最適化等) はデモ後。
- Awkernel 版 Executor の初版: ワーカタスクを `smp - 1` 個 (上限は設定可能) 起動し、スピン→yield のハイブリッドで待機。
  起床遅延を計測し、効果が出ない場合はスピン比率を上げる。待機時の CPU 消費はデモでは許容する。

### 4.5 ユーザインタフェース

- 第 1 段階: ブート後に固定プロンプトを 1 回推論してコンソールへストリーム出力 (llm.asm の `prompt=` 相当)。
- 第 2 段階 (決定): `awkernel_shell` に BLisp 関数 `(llm "...")` をフィーチャ付きで追加。FFI は要求をチャネルで LLM タスクへ渡すだけ (4.4 の S1)。
  マルチターン (KV キャッシュ保持)、履歴クリア、tok/s 表示を提供。生成中のプロンプト表示・中断の扱いは実装時に詰める。
- 発展: TCP 経由のチャット (QEMU hostfwd で `nc` から接続)。

### 4.6 KVM での実行

Makefile に LLM 用ターゲット (例 `qemu-x86_64-kvm`) を追加:
`-enable-kvm -cpu host -m 8G -smp N -nographic`、NUMA 分割は単純化。既存ターゲットは変更しない。

## 5. 段階的な進め方

段階 0–7 がデモの範囲。各段階は前段の完了条件を満たしてから進む。

| 段階 | 内容 | 完了条件 |
|---|---|---|
| 0. 下準備 | 現行 Awkernel が KVM (`-cpu host`) で起動するか確認。`nasm` 導入、llm.asm Phase 1 をホストでビルド、Qwen3-0.6B 取得と `.asmbin` 生成 (モデルはリポジトリに入れない) | KVM で既存シェルが動く。ホストで llm.asm が応答を返し、正解トークン列を採取できる |
| 1. エンジン (スカラ) | エンジンクレート `awkernel_llm` を新設: `.asmbin` パーサ、トークナイザ、チャットテンプレート、スカラ版 forward (行範囲 API + Executor 抽象)、サンプリング | ホストの `cargo test` と小さな CLI (example) で応答生成。greedy のトークン列が llm.asm と一致 |
| 2. SIMD 核 | AVX2/FMA/F16C 版 GEMV (FAST_Q8 相当)、RMSNorm、RoPE、softmax 等。スカラ版との差分テスト。`std::thread` の Executor で並列時のビット同一性も確認 | ホスト単体テスト通過。1 スレッドで llm.asm の 1T に近い tok/s |
| 3. カーネル基盤 | XCR0/OSXSAVE 有効化、IRQ 255 ハンドラの XSAVE、`test_avx_preempt`、ramdisk 対応 (`x86bootdisk` + カーネル公開 API)、KVM 用 make ターゲット | KVM 上で: AVX 命令が全 CPU で動く、プリエンプションテストが通る、モデルのヘッダを読める |
| 4. KVM 上で推論 | アプリクレート `applications/local_llm`: 専用 LLM タスク 1 つ (インライン Executor) で固定プロンプトを推論 | KVM 上でホスト版と同一のトークン列を出力 |
| 5. 対話 UI | シェルに `(llm "...")` を追加、マルチターン、履歴クリア、tok/s 表示 | シリアルから対話できる (シングルタスク版デモ) |
| 6. 並列化 | ワーカタスクによる Executor。ワーカ数によらない出力の同一性確認、スケーリング計測 | ワーカ数によらず同一出力。シングルより明確に速い |
| 7. デモ仕上げ | 手順書 (モデル準備〜起動)、既定パラメータ調整 | 手順書どおりに第三者が KVM でデモを再現できる |
| 以降 (デモ後) | 2 MiB ページ、起床遅延の最適化、RT タスクとの共存評価、TCP チャット、upstream 化の検討 | — |

### 5.1 段階 0 の結果 (2026-11-26 完了)

作業ディレクトリ (リポジトリ外): `/home/ytakano/program/rust/awkernel_ytakano/llm_work/`

| パス | 内容 |
|---|---|
| `llm.asm/` | llm.asm のクローン (commit `48feccb`)。`build/asm-llm` をビルド済み (`FAST_Q8=1`) |
| `models/qwen3-0.6b-bf16/` | Hugging Face `Qwen/Qwen3-0.6B` の `config.json` / `tokenizer.json` / `model.safetensors` |
| `models/qwen3-0.6b-q8_0.asmbin` | 変換済みモデル (v2, FP16 スケール, 643,469,504 B) |
| `venv/` | `export_weights.py` 用の Python 仮想環境 (numpy) |
| `ref/` | 正解データ: `err_t{1,4,8}.txt` (PROMPT_TOKENS / GEN_TOKENS と速度)、`out_t*.txt` (出力文)、`logits_prefill.bin` (プロンプト処理後の logits, f32 × 151,936) |

- **KVM 起動**: 現行 Awkernel (`make x86_64 RELEASE=1`) は `-enable-kvm -cpu host -machine q35 -m 8G -smp 4` で起動し、
  シェルで `(task)` が動いた。Primary heap は 7958 MiB。AVX 有効化前でも `-cpu host` での起動に支障なし。
- **正解器**: プロンプト "Explain how a CPU cache works in 3 concise sentences." (25 トークン) に対し 53 トークン生成して EOS。
  1/4/8 スレッドでトークン列は同一。
- **この開発機 (Ryzen 9 5900HX) での llm.asm の速度**:

  | スレッド | プロンプト処理 | 生成 |
  |---:|---:|---:|
  | 1 | 39.2 tok/s | 30.9 tok/s |
  | 4 | 83.2 tok/s | 62.6 tok/s |
  | 8 | 80.4 tok/s | 60.2 tok/s |

  → 1 スレッドは当初想定 (約 20 tok/s) より速い。一方、並列化の伸びは **約 2 倍で 4 スレッドで飽和** (メモリ帯域律速)。
  4.4 の「4 コア 3.4 倍」は EPYC での作者計測であり、この機体では当てはまらない。Awkernel 版のワーカ数は 4 程度で十分。
- `~/.ovmfpath` はビルド時に書き換わるため、ビルド後に元の値 (隣の `awkernel/` チェックアウト) へ戻した。
  このチェックアウトの OVMF は `target/ovmf/x64/` にある。KVM 用 make ターゲット (段階 3) では `~/.ovmfpath` への依存を避ける。

### 5.2 段階 1–2 の結果 (2026-11-26 完了)

- エンジンクレート `awkernel_llm/` (no_std + alloc、Awkernel 非依存)。ホスト検証用 `examples/llm_cli.rs`、
  モデルを使う統合テスト `tests/model.rs` (`AWKERNEL_LLM_MODEL` 未設定ならスキップ)。`make test` / `cargo test_awkernel_llm` で実行。
- **完了条件の修正**: 「greedy トークン列が llm.asm と全一致」は達成不能と判明したため変更した。
  llm.asm (活性化 Q8)、llm.asm 同梱の NumPy 参照 (活性化 f32)、本エンジンの 3 者とも 17〜19 トークン目の僅差の箇所で分岐する
  (活性化量子化の丸め差が増幅されるため)。新しい条件は次の 3 つ:
  1. プロンプトのトークン列が完全一致
  2. 先頭 17 トークンが一致 (3 者共通の部分)
  3. 5 トークン入力後の logits の NumPy 参照に対する cos が llm.asm 以上 (llm.asm 0.99955、本エンジン scalar 0.99979 / AVX2 0.99980)
- AVX2 版の速度 (この開発機、1 スレッド): プロンプト処理 40.2 tok/s、生成 27.9 tok/s (llm.asm は 39.2 / 30.9)。
- 行・ヘッドの分割数によらず logits はビット同一 (テスト済み)。
- `Chat::step` は 1 回あたり最大 1 フォワード。ターンを閉じる `<|im_end|>\n` は次ターンのプレフィル先頭で処理する
  (KV キャッシュの内容は llm.asm と同じ。テスト `multi_turn_cache_matches_flat_sequence`)。
- コードレビューで見送った性能項目 (デモ後に検討): バッチ化プレフィル (プロンプト処理が重みを毎トークン全読みするため最大の改善余地)、
  KV キャッシュの f16 化 (ctx 2048 で 448 MiB → 224 MiB)、O(n log n) の BPE。

### 5.3 段階 3–7 の結果 (2026-11-26 完了)

- **AVX とプリエンプション** (`awkernel_lib/src/arch/x86_64/xsave.rs`): 起動時に全 CPU で CR4.OSXSAVE と XCR0 = x87|SSE|AVX を設定。
  IRQ 255 ハンドラで AVX 成分のみをタスクのスタックに XSAVE/XRSTOR (4.3.1 の (a') のとおり。`context_switch` は無変更)。
  `test_avx_preempt` (KVM): 1 ワーカで保持中プリエンプション 300/300 回・破損 0、4 ワーカで保持中の CPU 移動 337 回・破損 0 で PASS。
  YMM 退避を外した負の対照では破損 300 回 / 848 回で FAIL し、テストの検出力を確認した。
- **ramdisk**: `make x86_64 RAMDISK=<file>` → `x86bootdisk --ramdisk`。ブートローダは ramdisk を使用中領域として報告するためヒープに取られない。
  仮想アドレスは DMA/ヒープ領域と衝突しないよう `0x500_0000_0000` に固定。`awkernel_lib::ramdisk::get()` で `&'static [u8]` として参照 (ゼロコピー)。
  614 MiB のモデルを含むイメージでも UEFI ブートに問題はなかった。
- **KVM**: `make qemu-x86_64-kvm` (`KVM_SMP=5 KVM_MEM=8G`、OVMF はこのチェックアウトの `target/ovmf/x64`、`Ctrl-A x` で終了)。
- **アプリ** `applications/local_llm`: サービスタスクが 1 ステップ 1 フォワードで回答をシリアルへストリーム出力。
  シェルに `(llm "...")` / `(llm_clear)` / `(llm_think bool)`、Ctrl-C で中断 (2 回でシェル側の待機も解除)。
  `local_llm_selftest` で起動時に参照トークンと照合 (KVM で PASS)。
- **並列化**: ヘルパータスク (最大 3) がチケットの CAS でチャンクを取り合う方式。呼び出し側も処理するため、ヘルパーが動けなくても完了する。
  ヘルパーは回答中スピンし約 1 ms ごとに yield、非回答時はスリープ。KVM での速度 (Ryzen 9 5900HX):

  | vCPU | プロンプト処理 | 生成 |
  |---:|---:|---:|
  | 2 (ワーカ 1) | 38 tok/s | 27 tok/s |
  | 3 (ワーカ 2) | 64 tok/s | 47 tok/s |
  | 5 (ワーカ 4) | 74 tok/s | 56 tok/s |

  回答テキストは vCPU 数によらず同一。ホストの llm.asm (4 スレッド: 83 / 63 tok/s) に近い。

### 5.4 コードレビュー

機能単位ごとにサブエージェントでレビュー (正しさ + 計算量・メモリ) を行い、指摘を反映した:

| 機能単位 | 主な指摘と対応 |
|---|---|
| エンジン (段階 1–2) | サンプラの空候補パニック、プレフィル中キャンセルの巻き戻し、1 ステップ 1 フォワードの保証 (ターン終了トークンの繰り延べ)、16 ヘッドが 1 チャンクに集まる分割、GQA で KV を 2 回読む問題 (KV ヘッド単位で処理し KV 読み出しを半減)、マージ表の空きスロット検査 ほか |
| カーネル基盤 (段階 3) | テストの空振り PASS (保持中の中断・CPU 移動を実測して判定、TSC で時間ベース)、AVX2 の有無確認、XCR0 を全 CPU で同値に、XSAVE 領域のヘッダのみゼロ化・XSAVE 後の `vzeroupper`、ramdisk の配置アドレス固定、空 ramdisk の拒否 |
| アプリ・シェル (段階 4–5) | サービス異常終了時のシェル固着 (終了ガード + Ctrl-C 2 回)、Ctrl-C の取りこぼし、`[]` 表示、回答中の入力の保持、中断時の統計 |
| 並列実行器 (段階 6) | パニック時のクロージャ解放後使用 (ループ終了ガード)・ハング (チャンク完了ガード)、同時呼び出し、ヘルパーの CPU 占有 (時間ベース yield)、偽共有 |

## 6. リスク

- **プリエンプションと AVX 状態** (4.3)。誤るとサイレントに計算が壊れる。段階 3 で意図的に壊すテストを用意する。
- **並列化の起床遅延** (4.4)。Awkernel のタスク起床遅延は未計測。大きい場合、スピン主体にしないと並列化の効果が出ない。
- **ramdisk のサイズ**: 614 MiB をブートローダが扱えるか・起動時間が許容範囲かは未検証 (段階 3 の最初に確認、だめなら fw_cfg)。
- **4 KiB ページ**: 614 MiB の重みを毎トークン全走査するため TLB ミスが効く。llm.asm は 2 MiB ページで約 10% 改善と主張。
- **浮動小数の一致**: `expf`/`sincos` の実装差で llm.asm とビット一致しない可能性。一致基準は「greedy トークン一致」とする。
- **no_std の数学関数**: `libm` クレート (autoware アプリで使用実績あり) か自前実装。
- **CI**: 新クレートはフィーチャ付きオプション依存なので既存 CI には乗らない。ホスト単体テストを `make test` に足すかは要判断。
- **検証済みコンポーネント**: context switch や `task.rs` を触る場合は `specification/` の更新要否を確認。

## 7. 決定事項

- エンジンは **Rust 再実装**。Awkernel 非依存の**別クレート** (このワークスペース内、仮称 `awkernel_llm`) とし、アプリクレートから使う (4.1)。
- 対象モデルは **Qwen3-0.6B 固定**。形式は llm.asm の `.asmbin` を流用 (GGUF 等は対応しない)。
- モデルは **ブートローダの ramdisk** で渡す (4.2)。
- プリエンプション対策は **(a') IRQ 255 ハンドラ内のスタックローカル XSAVE** (4.3.1)。
- タスク構成は**専用 LLM タスク**。シェルタスク内では推論しない (4.4)。
- **並列化 (ワーカタスクによる行分割) をデモの範囲に含める**。順序はシングルタスクで一致確認 → 対話 UI → 並列化 (4.4, 5 章)。
- UI は **シェル統合 `(llm "...")`** (4.5)。
- 当面の狙いは **デモ (KVM 上で対話できること)**。性能チューニングとリアルタイム性の評価はデモの後。
- upstream へのマージはデモ後に検討。今は考慮しない。

## 8. 残課題 (デモ後)

- 性能: バッチ化プレフィル (プロンプト処理が重みを毎トークン全読み)、KV キャッシュの f16 化 (448 MiB → 224 MiB)、2 MiB ページ、
  O(n log n) の BPE、KVM の pause-loop exit の影響調査 (未確認)。
- リアルタイム性: ヘルパーはスピン待ちで CPU を使う。RT タスクとの共存評価時に待機方式 (起床 IPI 等) を再設計する。
- 既存コードの気付き (今回は未変更): IRQ 255 ハンドラは実行中タスクがないとき EOI 前に return する (`kernel/src/arch/x86_64/interrupt_handler.rs`)。
  APIC の in-service ビットが残る可能性があるので別途確認したい。
- モデル品質: 0.6B・Q8 のため日本語などの回答は誤りが多い。think モードは `</think>` で閉じずに終わることがある (llm.asm でも同様)。
- upstream 化の検討 (カーネル変更 = AVX 有効化・YMM 退避・ramdisk を先に分ける)。

## 9. 発展: LLM によるシェル操作 (ツール呼び出し、未着手)

LLM が Awkernel のシェルコマンドを実行し、その結果をもとに回答する。Qwen3 のツール呼び出し形式
(`<tool_call>` 151657 / `</tool_call>` 151658 / `<tool_response>` 151665 / `</tool_response>` 151666) を使う。

### 9.1 流れ

1. `(llm "今動いているタスクは？")` を受けたら、ツール定義を含むシステムプロンプトを付けて生成する。
2. モデルが `<tool_call>{"name": "task", "arguments": {}}</tool_call>` を出したら、`</tool_call>` で生成を止める。
3. JSON を解析し、許可リストにあるコマンドだけをシェルタスクで実行して出力を受け取る。
4. 出力 (長さを制限) を `<tool_response>…</tool_response>` として渡し、生成を再開する。1 質問あたりの呼び出し回数に上限を設ける。

### 9.2 実行してよいコマンド (決定)

- 許可: 読み取り専用の `(task)`、`(interrupt)`、`(ifconfig)`。`perf` フィーチャ有効時は `(perf)` も候補。
- **除外: `(reboot)`、`(shutdown)`、`(netdump id)`。** ユーザー確認があってもツールからは実行できないようにする
  (ツール定義に載せず、実行側の許可リストにも入れない)。
- `(llm ...)` 系 (再帰) と任意の BLisp 式の評価も対象外。

### 9.3 必要な変更

- エンジン: ツール定義入りシステムプロンプト、`</tool_call>` での停止 (`StopReason::ToolCall` 相当)、ツール結果のターン追加。
- コマンド出力の取得: 当面はコンソール出力のキャプチャ方式 (既存コマンドを変えずに済む)。
  将来はコマンドを文字列を返す関数に整理する。
- 実行場所: BLisp の評価環境はシェルタスクにあり、`local_llm` はシェルに依存できない (依存が逆向き) ため、
  サービスからシェルへ「ツール実行要求」を送り、シェルの `is_busy()` 待ちループ内で実行して結果を返す。
- no_std の小さな JSON パーサ (`{"name": ..., "arguments": {...}}` のみ)。

### 9.4 リスクと対策

- 0.6B の呼び出し精度: 形式崩れや存在しないツール名。`<tool_call>` 直後は許可したツール名しか出せないよう
  サンプリングを制約する、think モードを切る、解析失敗時はエラーを返して 1 回だけ再試行させる。
  不十分なら Qwen3-1.7B を検討 (Q8 で約 2 GB、速度は約 1/3。互換性は未確認)。
- 速度と文脈長: コマンド出力は数百トークンになり、プロンプト処理 (約 74 tok/s) で数秒かかる。出力を切り詰める。
  バッチ化プレフィルの効果が大きい。

### 9.5 進め方

1. ホストの `llm_cli` で、0.6B がツール呼び出しを正しく出す率を測る (ツール名の制約あり/なし)。実装前に評価だけ先に行う。
2. 結果を見て、制約付きサンプリングかモデル大型化かを決める。
3. キャプチャ方式と 3 つの読み取り専用ツールで試作し、KVM で確認する。
