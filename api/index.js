// ============================================================
//  BMC Control-M — UI Server
//  Run: node index.js
//  Then open: http://localhost:3000
// ============================================================

const express = require("express");
const http = require("http");
const path = require("path");
const { requestLogger, notFound, errorHandler } = require("./src/middleware");

const app = express();
const publicDir = path.join(__dirname, "public");
const AGENT_BACKEND = process.env.AGENT_URL || "http://127.0.0.1:5001";

app.use(express.json());
app.use(requestLogger);

app.get("/health", (_req, res) => {
  res.json({ ok: true, service: "bmc-control-m-ui", agent: AGENT_BACKEND });
});

// Proxy chat to Python agent (avoids CORS; works on Windows/macOS/Linux)
app.post("/api/chat", (req, res) => {
  req.setTimeout(600000); // 10 minutes
  const bodyData = JSON.stringify(req.body);
  const targetUrl = new URL(`${AGENT_BACKEND}/api/chat`);

  const options = {
    hostname: targetUrl.hostname,
    port: targetUrl.port,
    path: targetUrl.pathname,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(bodyData),
      "Accept": "application/json"
    },
    timeout: 600000 // 10 minutes timeout
  };

  const proxyReq = http.request(options, (proxyRes) => {
    let responseData = "";
    proxyRes.on("data", (chunk) => {
      responseData += chunk;
    });
    proxyRes.on("end", () => {
      res.status(proxyRes.statusCode || 200);
      try {
        const data = JSON.parse(responseData);
        res.json(data);
      } catch (e) {
        res.send(responseData);
      }
    });
  });

  proxyReq.on("error", (err) => {
    res.status(502).json({
      session_id: req.body?.session_id,
      response:
        "Could not reach the automation agent. In another terminal run: npm run start:agent",
      error: err.message,
    });
  });

  proxyReq.on("timeout", () => {
    proxyReq.destroy(new Error("Timeout reaching the agent"));
  });

  proxyReq.write(bodyData);
  proxyReq.end();
});

app.use(express.static(publicDir));

app.use(notFound);
app.use(errorHandler);

const PORT = process.env.PORT || 3000;
const server = app.listen(PORT, () => {
  console.log(`\nBMC Control-M UI running on http://localhost:${PORT}`);
  console.log(`Chat proxy → ${AGENT_BACKEND}/api/chat`);
  console.log("Keep this terminal open. Press Ctrl+C to stop.\n");
});
server.timeout = 600000;
server.keepAliveTimeout = 600000;
server.headersTimeout = 610000;
server.requestTimeout = 600000;
