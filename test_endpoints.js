const http = require('http');

function postJson(url, data) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);
    const body = JSON.stringify(data);
    const req = http.request({
      hostname: parsedUrl.hostname,
      port: parsedUrl.port,
      path: parsedUrl.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body)
      }
    }, (res) => {
      let responseBody = '';
      res.on('data', (chunk) => { responseBody += chunk; });
      res.on('end', () => {
        resolve({
          statusCode: res.statusCode,
          headers: res.headers,
          body: responseBody
        });
      });
    });

    req.on('error', (err) => reject(err));
    req.write(body);
    req.end();
  });
}

async function run() {
  console.log("--- Testing UI Proxy (port 3000) with 'get connection profiles' ---");
  const t0 = Date.now();
  try {
    const proxyRes = await postJson('http://127.0.0.1:3000/api/chat', { message: 'get connection profiles' });
    console.log(`Status: ${proxyRes.statusCode}`);
    console.log(`Body: ${proxyRes.body}`);
    console.log(`Completed in ${((Date.now() - t0)/1000).toFixed(2)} seconds`);
  } catch (err) {
    console.error(`Proxy error: ${err.message}`);
  }
}

run();
