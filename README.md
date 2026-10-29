# ClientBench 模型效能與 Agent 能力比較工具

ClientBench 比較 `gpt-6.1-sol`、`gpt-6-luna`、Nous Router 的雲端模型，以及 llama.cpp `models` 資料夾內的 GGUF 模型。工具自動探索本機安裝目錄，逐一載入模型，使用固定版本的 prompts，輸出 HTML、CSV 和完整執行紀錄。新增 4B、8B、14B 或 35B 模型後，下一次執行會重新建立模型清單，不必逐個填 model ID。

這是可執行的 Python 3.12 原型，只有 Python 標準函式庫依賴。LLM 與重播 prompts 固定取自官方 MLPerf Client `v2.0.1`，commit `3cbc120e753d06b10c4ddbe53b99c461d83c8248`。模型、API、計時邊界和自主 agent 評分是本工具的設計，結果應標示 **MLPerf Client inspired**，不能稱為官方或認證 MLPerf 成績。

ClientBench v0.3.0 更新至 MLPerf Client **2.0.1** base。與先前 2.0.0 bundle 比較，所使用的 Llama 3.1 LLM prompts、Agent 重播對話與案例數沒有變更；兩個 Agent 輔助腳本只有換行格式差異，仍以新 commit 的原始位元組與 SHA-256 打包。基礎 LLM 13 案例、`--extended` LLM 15 案例、extended 固定重播 2 案例、自主 Agent 6 案例。現有指令不變；歷史 run 的來源版本保持不變。每份報告會直接顯示該次 run 的 MLPerf base 與 commit。

## 開始使用

### 模擬官方 MLPerf Client：LLM 與 Agent 指令對照

官方 MLPerf Client v2.0.1 的 Agent（SWE、Data Analyst）是**固定對話重播**：每一步把官方參考對話送進模型，再執行參考回覆裡寫好的工具呼叫（`src/CIL/llm/txt2txt_executor.cpp` 的 `RunTools(expected_output_prompt)`），模型自己的輸出不會決定下一步。工具直接在本機以 `cmd.exe`／`popen` 執行，Python 用隨附 portable Python 或 `--python-path`，**不使用 Docker**（`src/CIL/llm/llm_tools.cpp`）。

ClientBench 對應如下。除了標示「需要 Docker」的那一列，都不需要 Docker：

| 官方 MLPerf Client | ClientBench 指令 | 說明 |
|---|---|---|
| LLM 基礎工作負載 | `--suite llm` | 13 案例 |
| LLM 含 8K 摘要 | `--suite llm --extended` | 加入 Summarization Substantial（8K），15 案例 |
| SWE Agent | `--suite replay --replay-tools` | 3 個推論步驟，執行官方固定工具 |
| SWE + Data Analyst Agent | `--suite replay --extended --replay-tools` | 加入 Data Analyst 4 步驟 |
| （ClientBench 額外）LLM + 官方 Agent + 自主 agent | `--suite all --extended --replay-tools` | **需要 Docker**，因為包含自主 agent，見下方說明 |
| `--python-path <dir>` | `--replay-python <python 路徑>` | 預設使用執行 `bench.py` 的 Python |

官方的 `-q base|ext|both` 由 config 中各 scenario 的 prompt 集合決定；ClientBench 的 `--extended` 是本工具自己的分組（加入 8K 摘要與 Data Analyst），不保證與官方 `ext` 集合一一對應。要涵蓋全部隨附官方 prompts，請使用 `--extended`。

```powershell
# llama.cpp 安裝根目錄（內含 llama-server 與 models 子目錄）；換成自己的路徑
$LLAMA = "<LLAMA_CPP_ROOT>"

# 1) 官方 LLM suite（extended）
python bench.py run --target local --llama-root $LLAMA --suite llm --extended

# 2) 官方 Agent suite（SWE + Data Analyst，固定重播 + 執行官方工具）
python bench.py run --target local --llama-root $LLAMA --suite replay --extended --replay-tools

# 先檢查計畫，或縮短第一輪
python bench.py run --target local --llama-root $LLAMA --suite replay --extended --replay-tools --dry-run
python bench.py run --target local --llama-root $LLAMA --suite replay --extended --replay-tools --repeats 1 --warmups 0
```

> **`--suite agent` 不是官方 Agent suite。** 它是 ClientBench 自己加的自主 agent 測試：模型自己決定工具呼叫，`execute` 會執行**模型產生的程式碼**，因此必須在 Docker 隔離環境中跑（先 `docker pull python:3.12-slim`）。沒有 Docker 時會在載入模型前就以 `docker_not_found` 停止。`--suite all` 也包含這些自主案例；只要模擬官方，請分開執行上面的 `llm` 與 `replay`。

與官方仍有差異：ClientBench 透過各模型自己的 chat template／API 傳送同一份文字，不重現官方 C++ token 拼接、特定硬體 backend、expected-duration 換算或官方評分公式。工具只允許 `python …` 命令（官方 `execute` 可跑任意 shell），`patch` 也是單次文字替換，而不是 `git apply`；官方參考呼叫只用到 `read` 和 `python …`，所以不影響重播。結果請標示 **MLPerf Client inspired**。

### 指定 GPU：AMD／NVIDIA／Intel

`--local-device` 會先用 `llama-server --list-devices` 列舉裝置，只有**恰好一個**符合時才選用；找不到或有多個符合時會停止並列出可用裝置，請改用實際 ID（例如 `CUDA0`、`Vulkan1`、`SYCL0`）。它會固定單一裝置、關閉自動 fit 調整並保存裝置列舉結果；實際 GPU offload 仍需核對 `servers/` log。不會猜測 Vulkan 索引。

| 值 | 選擇規則 |
|---|---|
| `amd` | 名稱含 AMD 與 Radeon；不區分內顯與獨顯 |
| `nvidia` | 名稱含 NVIDIA |
| `intel` | 名稱含 Intel；Arc 獨顯與內顯都算，同機有兩者時請用實際 ID |
| 實際 ID | 例如 `CUDA0`、`Vulkan0`、`SYCL0`，比對 `--list-devices` 的 ID |

要用哪種 llama.cpp 預編譯套件取決於 GPU：NVIDIA 用 CUDA 版（也可用 Vulkan 版）；AMD 用 Vulkan 版（或 HIP 版）；Intel 用 SYCL 版或 Vulkan 版。先確認該套件看得到裝置：

```powershell
& "<LLAMA_CPP_ROOT>\llama-server.exe" --list-devices
```

```powershell
# AMD（Vulkan 或 HIP 版套件）
python bench.py run --target local --llama-root "<LLAMA_CPP_ROOT>" --local-device amd --gpu-layers 99 --suite llm --extended

# NVIDIA（CUDA 版套件；同機有多張卡時改用 --local-device CUDA0）
python bench.py run --target local --llama-root "<LLAMA_CPP_ROOT>" --local-device nvidia --gpu-layers 99 --suite llm --extended

# Intel（SYCL 或 Vulkan 版套件；內顯與 Arc 並存時改用 --local-device SYCL0 或 Vulkan0）
python bench.py run --target local --llama-root "<LLAMA_CPP_ROOT>" --local-device intel --gpu-layers 99 --suite llm --extended
```

Linux／macOS 把 `llama-server.exe` 換成 `llama-server`，路徑改用 `/path/to/llama.cpp`。官方 Agent 重播用同樣的 `--local-device` 等旗標加上 `--suite replay --extended --replay-tools`。

加 `--dry-run` 可先檢查 15 個案例、裝置及逐案例 context 設定；加 `--repeats 1 --warmups 0` 可縮短第一輪測試。GGUF 須放在該安裝根目錄的 `models` 子目錄。預設讀取各官方 prompt 的 `model_config.context_length`；`--context-size` 可明確指定所有案例共用的容量，優先於逐模型設定，但這會偏離官方逐工作負載的容量配置。`--gpu-layers` 也優先於逐模型設定。使用 `--local-device` 時，請移除 `server_args` 中重複的 device、split-mode、fit 與 GPU layers 旗標。外部已啟動 server 無法用這些旗標重新設定裝置或 context，執行計畫與結果會標為 `external_server`，實際容量留空。


解壓後在本資料夾開啟終端機。需要 Python 3.12 以上，以及已編譯的 llama.cpp／llama-server。LLM 與官方 Agent 重播（含 `--replay-tools`）只需要本機 Python；只有 ClientBench 自主 agent（`--suite agent`）需要 Docker 與 `python:3.12-slim` 映像。工具不會自動安裝、下載模型或拉取映像。

```powershell
python bench.py discover
python bench.py run --target local --suite llm --dry-run
python bench.py run --target local --suite llm
```

第一行只查找安裝目錄及 GGUF。第二行顯示執行計畫。第三行依序測試所有找到的模型，預設每個案例 warmup 一次、正式測量三次。

若 llama.cpp 放在較深、特殊或無法自動存取的位置，可以指定安裝根目錄；之後也能把它存到設定檔的 `llama_roots`。

```powershell
python bench.py discover --llama-root "<LLAMA_CPP_ROOT>"
python bench.py run --target local --llama-root "<LLAMA_CPP_ROOT>" --suite llm --extended
```

`--llama-root` 可重複指定多個安裝目錄。`--search-root "<PARENT_DIR>"` 則從該目錄探索子目錄。`--model-filter 35B` 可選擇檔名含該字串的模型；未指定時會測試全部模型，**不會用檔名猜測參數量並排除其他模型**。

雲端比較需要自己的 OpenAI API key，放在 `OPENAI_API_KEY` 環境變數。設定檔只保存環境變數名稱，不保存 key。`gpt-6.0 luna` 對應的官方 API 名稱是 `gpt-6-luna`。[Luna 官方文件](https://developers.openai.com/api/docs/models/gpt-6-luna)、[Sol 官方文件](https://developers.openai.com/api/docs/models/gpt-6.1-sol)。

```powershell
python bench.py run --target sol --target luna --suite llm
python bench.py run --suite all --extended
```

### Nous Plus：同一把 API key 切換雲端模型（v0.2.0）

在同一個 PowerShell 視窗設定 `NOUS_API_KEY`，再查詢 Nous 的即時模型清單。API key 不放進 JSON 設定檔，也不會記錄到 reports 或 traces。

```powershell
$env:NOUS_API_KEY = "YOUR_NOUS_API_KEY"
python bench.py models --target nous
python bench.py models --target nous --filter flash
```

清單只做模型查詢，不發送推論。請複製清單第一欄的完整 ID；官方產品名稱、直連廠商的 model ID，以及 Router ID 不一定相同。`--json` 顯示 context、工具能力及可用價格資料；`--out nous-models.json` 可保存清單。

```powershell
# Run one model; replace the placeholder with an ID from the catalog.
python bench.py run --target nous --cloud-model "MODEL_ID_FROM_LIST" --suite llm --extended --repeats 1 --warmups 0

# Compare two models with the same NOUS_API_KEY.
python bench.py run --target nous --cloud-model "FIRST_MODEL_ID" --cloud-model "SECOND_MODEL_ID" --suite llm --extended

# Compare a Nous model with all discovered local GGUF models.
python bench.py run --target nous --target local --cloud-model "MODEL_ID_FROM_LIST" --suite llm --extended
```

`--cloud-model` 可以重複使用。每個 ID 會建立獨立 target，依序執行並合併到同一份報告；單一 agent 任務不會中途換模型。未指定 ID 時會提示先選模型，不會自動跑整個 catalog。`--dry-run` 只列出計畫，不需要 API key、不查網路；正式執行先查詢 catalog，確認 ID 再開始推論。`--mock` 使用合成回覆，亦不使用任何 key 或網路。

Nous target 採 OpenAI-compatible Chat Completions，base URL 為 `https://inference-api.nousresearch.com/v1`，以 `Authorization: Bearer` 傳送 `NOUS_API_KEY`。它不帶 llama.cpp 的 template／cache／seed 參數，亦不強制 temperature 或特定 provider 的 reasoning 參數；各雲端模型使用其預設設定。若要固定支援的特定參數，可在 Nous target 的 `extra_body` 明確設定，並獨立成批比較。[官方 Nous 整合與 endpoint](https://hermes-agent.nousresearch.com/docs/integrations/nous-portal)、[官方代理的 Chat Completions contract](https://hermes-agent.nousresearch.com/docs/user-guide/features/subscription-proxy)。

Nous 的生成上限預設 8192 tokens，包含 reasoning，避免把本機 128–512 token 上限直接套在 thinking 雲端模型。若仍達到上限，可調整 Nous target 的 `max_output_tokens`。回覆 model ID 和 provider（若服務回報）保存在 `resolved_model`／`resolved_provider`；router 的服務排隊、轉送與上游推論都包含在測量時間內。路由提供者可能隨 Nous 配置改變，結果代表該次部署路徑。

自主 agent 用 `--suite agent`，需要 Docker（官方式 Agent 重播 `--suite replay --replay-tools` 不需要）。catalog 明確不支援 tools 的模型會在執行前拒絕；省略能力資料的模型標示 unknown，不能當成已驗證支援。模型需支援一致的工具 schema；工具呼叫與 reasoning history 都保存在 trace。

Nous 是 **opt-in** target，只有 `--target nous` 或手動啟用設定才執行。舊設定檔會在載入時加入預設 Nous 模板；不需要覆蓋既有 GPU、GGUF 或 OpenAI 設定。可在 `config.example.json` 的 Nous target 填入固定 `model`，或以 `--cloud-model` 當次覆寫。Plus 推論消耗 Nous credits；價格預設未知，不把它當作免費或自行猜測費率。`max_requests` 仍是每個模型的請求數限制。API key 遭拒時，模型清單指令回報 HTTP status；請核對 key 權限及帳戶可用模型。正式 Nous key 與推論尚需在帳戶環境實測。

若只想驗證工具、不要發送 API 請求或載入模型：

```powershell
python bench.py run --mock --extended --repeats 1 --out demo-new
```

MOCK 的回覆、計時、模型名稱和工具執行都是合成示例。隨附 `demo/report.html` 僅供查看介面，不是模型測量結果。

## 自動尋找與逐一載入 GGUF

支援的典型目錄如下。`models` 之下可以再有模型家族或量化格式子目錄。

```text
llama.cpp/
  llama-server.exe                   ← 預編譯套件也可放這裡
  build/bin/Release/llama-server.exe ← 或 CMake Windows build
  build/bin/llama-server             ← 或 Linux build
  models/
    model-4B-Q4_K_M.gguf
    family/model-35B-Q5_K_M.gguf
```

探索優先使用明確指定的 `llama_roots`，再使用 `search_roots`，最後使用預設搜尋：目前工作目錄、使用者 Documents／Desktop／Downloads／home、PATH 中的 llama-server，以及固定磁碟的有限深度目錄。搜尋有深度及 20,000 個目錄上限，避開系統目錄、大型快取和符號連結；不是全磁碟無限搜尋。無權限的位置會出現在 inventory 的 warnings。若 server 放在自訂位置，移至上述結構或指定適合的安裝根目錄。

每次執行都重新列出 `models` 下的 `.gguf`，排序後將每個檔案建立為獨立 target。不同量化檔案各有自己的結果。`*-00001-of-000NN.gguf` 是分片模型入口，其他分片不重複啟動；缺少分片會記錄載入失敗。以 `mmproj`／`projector` 開頭的 GGUF 列為投影器輔助檔案，不當作文字模型。其他無法作為文字模型的 GGUF 會留下載入失敗紀錄。

每個模型使用新的 localhost port 和 server process，等待 `/health` 就緒後執行所有案例，最後關閉再載入下一個模型。測試不會把多個 35B 模型一起放入記憶體。載入失敗或逾時會保留結果並繼續其他模型。工具關閉的是自己啟動的程序；`discover_models:false` 可改接已啟動的 server，這種模式不負責其程序生命週期。

記錄包括模型檔案與各分片 SHA-256、檔案大小、server binary SHA-256、啟動參數、`/props`、載入時間和 server log。雜湊計算可能花時間，但不納入推論延遲。模型載入時間另列，亦不併入已載入模型的 TTFT。

## 三條獨立測試路徑

| 路徑 | 使用的輸入 | 衡量問題 | 品質判定 |
|---|---|---|---|
| `llm` | 官方 v2.0.1 原始 prompts | 相同文字工作負載的反應與生成速度 | CSV 轉換核對值；XML 只驗證語法；其他保留待評分 |
| `replay` | 官方 SWE／Data Analyst 固定對話與參考回覆 | 固定 agent 工作流程的推論成本與工具時間 | 不給自主任務成功率 |
| `agent` | 依相同情境建立的可驗證任務 | 模型是否會讀檔、修改、執行並完成任務 | SWE 隱藏測試；資料分析數值 oracle |

v2.0.1 原始碼的工具執行使用 `expected_output_prompt`，後續 context 也沿用參考對話。因此本工具將「固定重播」和「自主 agent」分開：重播接入原始對話文字，生成結果只做紀錄；自主 agent 會真的依模型選擇的工具與參數執行下一步。[固定版本執行器原始碼](https://github.com/mlcommons/mlperf_client/blob/3cbc120e753d06b10c4ddbe53b99c461d83c8248/src/CIL/llm/txt2txt_executor.cpp)。

### 官方 LLM 工作負載

| Scenario | 官方概略輸入／輸出 tokens | 隨附案例數 | 預設啟用 |
|---|---:|---:|---|
| Content Generation | 128／256 | 3 | 是 |
| Creative Writing | 512／512 | 3 | 是 |
| Structured Text | 830／256 | 2 | 是 |
| Code Analysis | 1856／128 | 3 | 是 |
| Summarization Intermediate | 3906／128 | 2 | 是 |
| Summarization Substantial | 7631／128 | 2 | `--extended` |

以上概略長度是 MLCommons 的工作負載描述，不是對所有 tokenizer 保證的長度。工具保留同一份原始 system／user 文字，由每個模型自己的 chat template 格式化；不把 Llama 特殊 token 直接送入 GPT。[MLCommons 工作負載說明](https://mlcommons.org/benchmarks/client/)、[官方 prompt 格式](https://github.com/mlcommons/mlperf_client/blob/3cbc120e753d06b10c4ddbe53b99c461d83c8248/docs/LLM_PROMPTS_GENERATION.md)。

### 固定 agent 重播

SWE 有三個推論步驟，Data Analyst 有四個。`--suite replay` 只執行 SWE；`--extended` 加入 Data Analyst。預設只測推論重播，報告標示「僅推論重播」。

```powershell
python bench.py run --suite replay --extended --replay-tools
```

`--replay-tools` 在每次推論後執行官方參考回覆中的固定工具。隨附上游 Python 工具及完整 CSV 資料，無須另外下載。與官方相同，這些固定工具直接用本機 Python 執行，不需要 Docker；cwd 是該次的 `workspaces/` 目錄，只允許 `python …` 命令。預設用執行 `bench.py` 的 Python，可用 `--replay-python` 指定（對應官方 `--python-path`）；實際 interpreter 版本與路徑記錄在結果的 `tool_runtime`。找不到時會在載入模型前停止。工具輸出會保存供查核，後續 prompt 仍使用原始固定對話。這是跨 provider 的可攜重播，未重現官方 C++ token 拼接、特定硬體 backend 或 expected-duration 換算，不應拿其數字直接取代官方成績。

### 自主 agent

每種情境有三個固定 seed。這六個小案例是功能驗證與初始比較集，規模不足以代表一般 agent 能力。

SWE 任務要求修正 bootstrap mean confidence interval：輸入驗證、百分位線性插值、單樣本／單次 resample 和可重現性。評分端在獨立唯讀 volume 提供八組隱藏測試，並要求模型自己有成功執行驗證的紀錄。

Data Analyst 任務要求讀取含重複、空值、負值與非數字的 CSV，產生並執行 `analysis.py`，寫入 `report.json`。評分端核對去重與排除規則、各供應商與各月份總量、最高供應商，不依模型的「已完成」宣稱給分。

```powershell
docker pull python:3.12-slim
python bench.py run --suite agent
```

三個後端共用 `read`、`write`、`patch`、`execute` schema。可恢復的工具錯誤會回給模型；超出回合或時間限制則失敗。`patch` 是單次精確文字替換，`execute` 只接受 Python 命令，沒有通用 shell。此設計限制了工具範圍，不能解讀為完整 SWE-bench 或一般網頁／桌面 agent 測試。

Python 執行用相同 Docker image ID、CPU、memory 和程序數限制，關閉網路、唯讀 container root，只掛載該次任務的 workspace。container 啟動與工具執行時間都算進 tool time；評分端自己的測試時間另列，不算進 agent E2E。若 Docker 或映像不可用，會在載入任何模型前以 `docker_not_found`／`docker_image_not_available` 停止，不改在 host 執行模型產生的程式。

## 計時與公平比較

`first_event_s` 是第一個 SSE 事件；`ttft_visible_s` 是第一個可見文字 delta；`ttft_tool_s` 是第一個工具呼叫片段。metadata／隱藏 reasoning 不算可見 TTFT。單個串流 delta 可能包含多個 tokens，因此本工具不把收到的 chunk 數當 token 數。

```text
Agent E2E = 所有模型請求時間 + 所有工具時間 + harness overhead
Native TPS estimate = (可見 native tokens − 1) / (末個可見 delta 時間 − 首個可見 delta 時間)
```

TPS 使用 provider usage 的原生 token 數。只有沒有 tool calls、且可分離 reasoning 的文字輸出才計算。usage 缺失、只有一個片段、或無法分离 reasoning，TPS 会留空。因為片段邊界不保證單 token，这只是估計；tokenizer 不同也会影响比较。`visible_chars_per_s` 提供同文字尺度的補充指標，使用完整 request 時間作分母，包含等待時間。

v0.2.1 報告新增 **Prefill p50 秒**、**Prefill TPS p50**，並將 Server TPS 改名為 **Generation TPS p50**。三者均為有效測量中各模型請求的中位數；Agent 的 prefill 秒數不是整個任務的合計。

- `prefill_s` = llama.cpp `timings.prompt_ms / 1000`，單位秒。
- `prefill_tps` = `timings.prompt_per_second`，單位 tokens/s。
- `generation_tps` = `timings.predicted_per_second`，包含 reasoning、可見文字與其他生成 tokens。
- `prefill_tokens` = `timings.prompt_n`，實際處理的輸入 tokens；`cached_prompt_tokens` = `timings.cache_n`，快取重用的 tokens。

Generation TPS 與可見 Native TPS 分欄顯示，不能解讀為只有最終答案的速度。`ttft_reasoning_s` 與 `ttft_visible_s` 也分開保存。雲端 API 未提供伺服器計時時，prefill 與 Generation TPS 顯示 `—`，不使用輸入 tokens / TTFT 代替。`server_decode_tps` 與彙整的 `server_decode_tps_p50` 保留為舊分析程式的相容欄位，數值與 generation 欄位相同。

舊測試可執行 `python bench.py report <run directory>`，從原有 traces 補取 prefill 計時，重新輸出 HTML、summary JSON 與 CSV，不會重跑推論或修改原始 `results.jsonl`。缺少 traces 或原始計時時仍顯示 `—`。

Sol 與 Luna 都使用 Responses API，預設 `reasoning_effort:low`；llama.cpp 使用串流 Chat Completions，預設 temperature 0。[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)、[llama.cpp server 文件](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)。Responses 的 `max_output_tokens` 包含隱藏推理，預設給 8192；本機 LLM 使用上游輸出目標當上限。**相同 visible 長度未被強制保證**，請同時查看實際輸出量、截斷狀態和品質，不能只拿較短回答的 E2E 判定模型較快。需做嚴格固定輸出長度 microbenchmark 時，應另建工作負載與共同 tokenization 規則。

雲端請求時間包含網路、服務排隊與推論；本機包含 HTTP 與推論，不能把差值直接當硬體效能差。cloud cached-input tokens 有回報就記錄；不能保證關閉雲端 cache。本機預設 `cache_prompt:false`，server 不做 context shift；warmup 與 model load 不計入正式測量。主結果是已載入模型的 sequential、concurrency 1 測試。多模型批次順序和案例順序由 seed 隨機化，同一模型內依 context 容量分組，所有容量組完成才換下一個模型，以減少重複載入成本；仍可能有時段與溫度偏差，正式實驗應重複整批並調換順序。

報告按 target／scenario／lane 分組，输出 p50、p95、完成次數、失敗數、品質檢查通過率、成功案例延遲與各 case median 的 bootstrap mean interval。warmup 不参与統計，失败不会从成功率分母移除。少量案例的 p95 和 interval 只能探索，不是穩定排名。

LLM 的正常 token 上限終止 (`finish_reason:length/incomplete`) 標示為 `truncated`。只要有文字或 reasoning 輸出，仍列入有效計時；完整完成次數與截斷次數分開呈現，截斷回覆不算品質通過。Agent 的截斷回合仍視為未完成任務，不列入成功延遲。

### Thinking 模型回覆空白的排查（v0.1.1 修正）

部分 Qwen 模型預設啟用 thinking，會先以 `reasoning_content` 生成推理。如果直接使用官方工作負載的 128–512 token 上限，可能尚未生成最終答案就達到上限。v0.1.1 預設在本機請求傳入 `chat_template_kwargs:{"enable_thinking":false}`，設定欄位為 `local_thinking:"off"`。不支援這個 template 選項的模型可能忽略它；請查看 trace 是否有 reasoning，以及 server 的 chat template。實際設定會記錄在結果與 run.json。[llama.cpp 官方參數說明](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)。

```powershell
python bench.py run --target local --suite llm --extended
python bench.py run --target local --suite llm --extended --local-thinking on
```

第一行使用預設的 thinking off。第二行明確啟用 thinking，單次生成上限改用 local target 的 `max_output_tokens`（預設 2048，含推理，且至少等於案例目標）。可在設定檔調高此上限；`--local-thinking auto` 保留模型 template 的預設，也使用這個較大的 budget。不同 thinking 設定應分成獨立批次比較。模型用完 budget 且只有 reasoning 時，報告提供 `reasoning_only_output_budget_exhausted` 診斷；可見 TTFT 留空，Generation TPS 仍可顯示。

不要在 `extra_body.chat_template_kwargs` 重複設定 `enable_thinking`，請使用 `local_thinking`；其他 template kwargs 會保留。

原型未整合完整 MMLU／IFEval、LLM judge、固定共同 tokenizer、GPU／RAM 取樣、能耗、負載併發或跨 host。這些屬於下一階段：先擴充可驗證的任務與獨立品質 gate，再加入資源量測及 concurrency profile；沒有執行的項目不會產生假分數。

## 4B 至 35B 的硬體設定

預設一次一個模型，context 跟隨官方 v2.0.1 各工作負載的設定，GPU layers 99、8 threads。這些是可調整的初始設定，不保證 35B 一定能放入你的 RAM／VRAM。記憶體不足會記錄該容量的載入失敗，仍會繼續其他容量，不暗中降低 context 或更換量化。

`context_size` 預設為 `null`（或省略），表示讀取原始 prompt 的 `model_config.context_length`。同官方程式，也接受舊式 `model_config.model.context_length`，兩者同時存在時以外層為準；未指定則使用 4096。

| 工作負載 | Context tokens |
|---|---:|
| 一般內容生成、創意寫作、結構轉換、intermediate4k | 4096 |
| 程式碼分析 | 5120 |
| substantial8k 長文本 | 8192 |
| SWE Agent 固定重播 | 12000 |
| Data Analyst Agent 固定重播 | 15000 |

本機 server 的 context 在啟動時決定，因此每個模型依容量分組，完成一組後關閉 server，再啟動下一個容量。容量組順序及組內案例順序由 seed 隨機化；同一組共用一次模型載入，所有重複測試完成後再換組。模型載入不計入推論與任務時間；warmup 仍逐案例執行。自主 agent 是客製任務，沒有對應的官方 prompt 設定，使用官方程式的 4096 預設值，可明確覆寫。

設定優先順序：`--context-size` → `model_overrides` 的數字 `context_size` → local target 的數字 `context_size` → 工作負載設定。若沿用舊的 `config.local.json`，請將其中固定的 32768（以及逐模型覆寫）改為 `null` 或刪除，並移除啟動指令的 `--context-size`，才會採用官方容量。

`plan.json` 會列出各案例的容量與來源；`results.jsonl`／`results.csv`／trace 保留 `workload_context_length`、`context_size` 與 `context_source`。`run.json` 的 `server_sessions` 記錄各容量的啟動參數、載入時間及 log。雲端模型的 context 由服務端決定，這些本機容量設定不會改變雲端模型。

例如在 `config.example.json` 的 local target 加入以下內容，key 使用 `models` 內的相對檔案路徑。省略 context 覆寫即可繼續跟隨官方工作負載：

```json
"model_overrides": {
  "small/model-4B-Q4_K_M.gguf": {
    "gpu_layers": 99
  },
  "large/model-35B-Q4_K_M.gguf": {
    "gpu_layers": 20,
    "load_timeout_s": 600
  }
}
```

比較參數量時，盡量讓量化、context、GPU offload 與工具環境一致；若不同，應將它們視為不同部署配置。`gpu_layers:0` 可明確測 CPU 模式。`server_args` 可指定 batch、KV cache type 等額外參數，但不得覆寫模型路徑、host、port、憑證、context 或 parallelism；要覆寫 context，請使用 `context_size` 或 `--context-size`。不同 server 版本的 flag 支援程度不同，失敗可從 log 查明。

## 設定與輸出

`config.example.json` 預設有三個 target 模板；local 會展開成每個 GGUF。`max_requests` 是**每個 target**上限，warmup 也計入，用來限制單一模型的請求數；某模型到達上限會留下部分執行提示，仍會測下一個模型。`max_agent_turns` 限定各任務回合數，`task_timeout_s` 限定任務，`request_timeout_s` 限定單次請求。無自動 retries。

價錢預設未知；若要計費估計，可為各 cloud target 設定 `price_per_million:{"input":...,"cached_input":...,"output":...}`，需自行填當次帳戶／region／service tier 的費率。報告記錄每次 usage，包含 warmup；未回報 cache 或 usage 時不猜金額。local 的硬體／電力成本未估，不當作零成本。

每次執行會建立新的 `runs/<UTC timestamp>/`，也可用 `--out` 指定新的空資料夾。

| 檔案 | 用途 |
|---|---|
| `report.html` | 可切換 lane、模型和搜尋字串的比較報告 |
| `results.jsonl`／`results.csv` | 逐次結果與 warmup |
| `summary.json`／`summary.csv` | 各模型各 scenario 統計 |
| `run.json` | 設定、case hashes、執行順序、環境與 model identity |
| `plan.json` | 執行前模型清單、案例數與最壞請求數估計 |
| `traces/` | 每回合實際輸入、回覆、usage、計時、工具參數與結果 |
| `servers/` | 各本機模型 server logs |
| `workspaces/` | 各 agent 任務產物 |

`python bench.py report <run directory>` 可重新產生報告。`python bench.py list --extended` 顯示完整案例。`--case-filter` 可只跑某個 prompt 或 seed。需要比較多個實驗批次時，先確認 `run.json` 的 case hashes 和部署參數一致，再合併分析，工具不會把不同設定的歷史結果自動混成一個分數。

## 原始碼結構與驗證

```text
bench.py → cli.py → local.py discovery/server lifecycle
                   runner.py scheduling/agent loop
                     transport.py Responses/llama.cpp SSE
                     workloads.py pinned prompts/task oracles
                     tools.py uniform tool environment
                   report.py CSV/HTML summaries
```

```powershell
python -m unittest discover -s tests -v
```

隨附 36 項測試覆蓋 SSE metadata／reasoning／工具片段、缺失或 null usage、不完整串流、stateless 工具歷史、模型探索／分片、server 清理、路徑邊界、確定性 oracle、warmup 排除、失敗分母與 request budget，並包含 thinking-only 截斷回覆的回歸測試、Nous 共用 key 多模型請求、catalog 驗證、cloud 請求參數與 key 不外洩檢查。`demo` 是完整離線流程驗證；真實 API、GGUF 與 Docker 的實測狀態記錄在交付的 verification.json，不能把 MOCK 當成真實成績。

## 授權與資料來源

本工具新增程式碼採 MIT；`data/mlperf_v2` 是 MLCommons 固定 commit 的原始檔案，保留其 Apache 2.0 license。`manifest.json` 記錄逐檔 SHA-256，執行前核對完整性。此資料夾內的 upstream prompts、scripts 與 CSV 沒有改寫；衍生 harness 與自主案例存放在 `clientbench/`。
