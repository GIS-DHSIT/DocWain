# DocWain UI — Streaming + Visualization Changes

**Target repo:** `https://github.com/GIS-DHSIT/docwain-ui` (branch: `develop`)

**Purpose:** Enable real-time streaming responses (first token in ~2s instead of 77s wait) and render chart visualizations from the backend.

---

## Backend Response Format

The backend `/api/ask` endpoint now streams text chunks followed by a trailing metadata block:

```
[text chunks arrive in real-time...]
## Invoice Expenses

| Category | Amount |
|----------|--------|
| Consulting | **$12,000** |
| Software | **$8,200** |

<!--DOCWAIN_MEDIA_JSON:{"media":[{"type":"chart","chart_type":"bar","title":"Expense Breakdown","png_base64":"iVBOR..."}],"sources":[...],"session_id":"session_abc","grounded":true,"context_found":true}-->
```

**Key points:**
- Text is streamed as plain text chunks (256 chars each, broken at natural boundaries)
- After all text, a `<!--DOCWAIN_MEDIA_JSON:{...}-->` block is appended
- The metadata JSON contains: `media`, `sources`, `session_id`, `grounded`, `context_found`
- `media` is an array of chart objects with `type`, `chart_type`, `title`, `png_base64`, `data_summary`
- The `<!--DOCWAIN_MEDIA_JSON:...-->` marker must be stripped before displaying text

---

## File Changes Required

### 1. `/src/services/api/api.ts` — Add Streaming Fetch Function

Add this new export alongside the existing `apiService` function:

```typescript
/**
 * Stream a POST request and invoke onChunk for each text chunk received.
 * After the stream completes, parses the trailing <!--DOCWAIN_MEDIA_JSON:...--> block.
 *
 * @returns The clean response text and parsed metadata (media, sources, session_id).
 */
export async function apiServiceStream(
  url: string,
  body: Record<string, unknown>,
  onChunk: (text: string) => void,
  token: string
): Promise<{ fullText: string; metadata: StreamMetadata }> {
  const baseUrl =
    import.meta.env.VITE_BASE_URL ||
    "https://docwain-api-g9crdtdyb8hncbaq.westeurope-01.azurewebsites.net/";

  const response = await fetch(`${baseUrl}${url}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Request failed: HTTP ${response.status}`);
  }
  if (!response.body) {
    throw new Error("Response body is empty");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";

  // Read stream chunks and forward to caller
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    fullText += chunk;
    onChunk(chunk);
  }

  // Parse trailing metadata block
  const mediaMarker = "<!--DOCWAIN_MEDIA_JSON:";
  const markerIdx = fullText.indexOf(mediaMarker);
  let metadata: StreamMetadata = {};
  let cleanText = fullText;

  if (markerIdx !== -1) {
    const jsonStart = markerIdx + mediaMarker.length;
    const jsonEnd = fullText.indexOf("-->", jsonStart);
    if (jsonEnd !== -1) {
      try {
        metadata = JSON.parse(fullText.substring(jsonStart, jsonEnd));
      } catch {
        // Ignore malformed metadata — text is still usable
      }
      cleanText = fullText.substring(0, markerIdx).trimEnd();
    }
  }

  // Also check X-Session-ID header
  const headerSessionId = response.headers.get("X-Session-ID");
  if (headerSessionId && !metadata.session_id) {
    metadata.session_id = headerSessionId;
  }

  return { fullText: cleanText, metadata };
}

/** Shape of the metadata block appended after streamed text. */
export interface StreamMetadata {
  media?: MediaItem[];
  sources?: Array<Record<string, unknown>>;
  session_id?: string;
  grounded?: boolean;
  context_found?: boolean;
}

/** A single visualization/chart returned by the backend. */
export interface MediaItem {
  type: string;          // "chart" | "chart_validation"
  chart_type?: string;   // "bar" | "donut" | "line" | "grouped_bar" | "radar" | "graph" | ...
  title?: string;        // Chart title
  png_base64?: string;   // Base64-encoded PNG image
  data_summary?: string; // Brief text summary of the chart data
}
```

---

### 2. `/src/services/selfassist/selfassist.ts` — Add Streaming Chat Function

Add a new streaming variant of `selfAssistChat`:

```typescript
import { apiServiceStream, StreamMetadata } from "../api/api";

/**
 * Streaming version of selfAssistChat.
 * Calls the backend with stream=true and forwards text chunks via onChunk.
 */
export async function selfAssistChatStream(
  prompt: string,
  profileId: string,
  email: string,
  subscriptionId: string,
  modelName: string,
  newSession: boolean,
  sessionId: string | null,
  agentMode: boolean,
  internetEnable: boolean,
  toolName: string | null,
  onChunk: (text: string) => void,
  token: string
): Promise<{ fullText: string; metadata: StreamMetadata }> {
  return apiServiceStream(
    API_PATHS.DAI.CHAT,
    {
      prompt,
      profile: profileId,
      email,
      subscription_id: subscriptionId,
      modelname: modelName,
      new_session: newSession,
      session_id: sessionId,
      stream: true,
      agentMode,
      enable_internet: internetEnable,
      toolName,
    },
    onChunk,
    token
  );
}
```

---

### 3. `/src/pages/Home/Home.tsx` — Update Message Interface

Add `media` and `isStreaming` to the message type. Find the message interface/type (or inline type) and update:

```typescript
interface ChatMessage {
  sender: "User" | "AI";
  text: string;
  timestamp: number;
  error?: boolean;
  isStreaming?: boolean;          // NEW: true while chunks are arriving
  media?: MediaItem[];            // NEW: charts/visualizations from backend
  sessionId?: string;             // NEW: session ID from metadata
}
```

---

### 4. `/src/pages/Home/Home.tsx` — Update `handleSubmit` for Streaming

Replace the existing `selfAssistChat` call block (~lines 210-240) with:

```typescript
// Add streaming AI message placeholder
setMessages((prev) => [
  ...prev,
  { sender: "User", text: userMessage, timestamp: Date.now() },
  { sender: "AI", text: "", timestamp: Date.now(), isStreaming: true },
]);

try {
  const { fullText, metadata } = await selfAssistChatStream(
    userMessage,
    selectedProfile,
    user.email,
    subscription_id,
    selectedModel,
    session,
    sessionId,
    agentMode,
    internetEnable,
    toolName,
    // onChunk — updates the last AI message in real-time
    (chunk: string) => {
      setMessages((prev) => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (lastMsg?.sender === "AI" && lastMsg.isStreaming) {
          // Strip any partial media marker from display
          let displayText = lastMsg.text + chunk;
          const markerIdx = displayText.indexOf("<!--DOCWAIN_MEDIA_JSON:");
          if (markerIdx !== -1) {
            displayText = displayText.substring(0, markerIdx).trimEnd();
          }
          updated[updated.length - 1] = { ...lastMsg, text: displayText };
        }
        return [...updated];
      });
    },
    token
  );

  // Stream complete — set final clean text + media
  setMessages((prev) => {
    const updated = [...prev];
    const lastMsg = updated[updated.length - 1];
    if (lastMsg?.sender === "AI") {
      updated[updated.length - 1] = {
        ...lastMsg,
        text: fullText,
        isStreaming: false,
        media: metadata.media || undefined,
        sessionId: metadata.session_id || sessionId,
      };
    }
    return [...updated];
  });

  // Update session ID if returned from backend
  if (metadata.session_id) {
    setSessionId(metadata.session_id);
  }
} catch (err: any) {
  // Replace the streaming placeholder with error message
  setMessages((prev) => {
    const updated = [...prev];
    if (updated[updated.length - 1]?.sender === "AI") {
      updated[updated.length - 1] = {
        sender: "AI",
        text: err.message || "Processing request failed.",
        timestamp: Date.now(),
        error: true,
      };
    }
    return [...updated];
  });
}
```

---

### 5. `/src/pages/Home/Home.tsx` — Add Chart Rendering in Message Display

Find where `<MarkdownRenderer content={msg.text} />` is rendered (~line 307) and add chart rendering immediately after it:

```tsx
{/* Existing markdown rendering */}
<MarkdownRenderer content={msg.text} />

{/* Streaming indicator */}
{msg.isStreaming && (
  <span style={{ display: "inline-block", marginLeft: 4 }}>
    <span className="streaming-cursor">|</span>
  </span>
)}

{/* Chart/Visualization rendering */}
{msg.media && msg.media.length > 0 && (
  <div style={{ marginTop: 16 }}>
    {msg.media.map((item, idx) =>
      item.png_base64 ? (
        <div
          key={idx}
          style={{
            marginBottom: 12,
            textAlign: "center",
            background: "#ffffff",
            borderRadius: 8,
            padding: 12,
          }}
        >
          {item.title && (
            <div
              style={{
                fontSize: 14,
                fontWeight: 600,
                color: "#333",
                marginBottom: 8,
              }}
            >
              {item.title}
            </div>
          )}
          <img
            src={`data:image/png;base64,${item.png_base64}`}
            alt={item.title || "Chart"}
            style={{
              maxWidth: "100%",
              height: "auto",
              borderRadius: 4,
            }}
          />
          {item.data_summary && (
            <div
              style={{
                fontSize: 12,
                color: "#666",
                marginTop: 6,
              }}
            >
              {item.data_summary}
            </div>
          )}
        </div>
      ) : null
    )}
  </div>
)}
```

---

### 6. Optional: Add Streaming Cursor CSS

Add to your global CSS or component styles:

```css
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

.streaming-cursor {
  animation: blink 0.8s step-end infinite;
  font-weight: bold;
  color: #90caf9;
}
```

---

## Testing Checklist

After applying these changes:

- [ ] **Streaming text**: Send a query — text should appear within 2-3 seconds, streaming word by word
- [ ] **Chart rendering**: Ask "compare the candidates' skills" or "show expenses breakdown" — a chart image should appear below the text response
- [ ] **No chart queries**: Ask "when does the contract expire?" — only text, no chart
- [ ] **Error handling**: Disconnect network mid-stream — error message should appear
- [ ] **Session continuity**: Follow-up questions should maintain session context
- [ ] **Media marker not visible**: The `<!--DOCWAIN_MEDIA_JSON:...-->` text should never be shown to the user

## API Response Examples

### Streaming response (text + chart):
```
Text chunks arrive...
## Expenses
| Category | Amount |
...
<!--DOCWAIN_MEDIA_JSON:{"media":[{"type":"chart","chart_type":"donut","title":"Expense Breakdown","png_base64":"iVBOR...","data_summary":"4 items, total $26,500"}],"sources":[{"name":"invoice.pdf"}],"session_id":"session_abc123","grounded":true,"context_found":true}-->
```

### Streaming response (text only, no chart):
```
Text chunks arrive...
The contract expires **January 1, 2027**.
<!--DOCWAIN_MEDIA_JSON:{"sources":[{"name":"contract.pdf"}],"session_id":"session_abc123","grounded":true,"context_found":true}-->
```

### Non-streaming JSON response (existing format, still works):
```json
{
  "answer": {
    "response": "The contract expires **January 1, 2027**.",
    "sources": [{"name": "contract.pdf"}],
    "grounded": true,
    "context_found": true,
    "media": [
      {
        "type": "chart",
        "chart_type": "donut",
        "title": "Expense Breakdown",
        "png_base64": "iVBOR...",
        "data_summary": "4 items, $26,500"
      }
    ]
  },
  "current_session_id": "session_abc123"
}
```
