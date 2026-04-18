import json


def build_player_html(video_url: str, frame_boxes: dict,
                      frame_w: int, frame_h: int, fps: float) -> str:
    boxes_json = json.dumps(frame_boxes)
    return f"""
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #000; }}

  #player-wrap {{
    position: relative;
    width: 100%;
    background: #000;
  }}
  #vid {{
    width: 100%;
    height: auto;
    display: block;
  }}
  #overlay {{
    position: absolute;
    top: 0; left: 0;
    pointer-events: none;
  }}
  #controls {{
    height: 44px;
    display: flex;
    align-items: center;
    padding: 4px 8px;
    background: #111;
  }}
  #toggleBtn {{
    padding: 5px 18px;
    font-size: 13px;
    font-weight: bold;
    border: 2px solid #00cc44;
    border-radius: 6px;
    background: #00cc44;
    color: #fff;
    cursor: pointer;
  }}
  #toggleBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
</style>

<div id="controls">
  <button id="toggleBtn" onclick="toggleBoxes()">Player Detection Boxes: ON</button>
</div>
<div id="player-wrap">
  <video id="vid" controls>
    <source src="{video_url}" type="video/mp4">
  </video>
  <canvas id="overlay"></canvas>
</div>

<script>
  const BOXES  = {boxes_json};
  const FPS    = {fps};
  const VID_W  = {frame_w};
  const VID_H  = {frame_h};
  let showBoxes = true;

  const vid    = document.getElementById('vid');
  const canvas = document.getElementById('overlay');
  const ctx    = canvas.getContext('2d');

  function resizeCanvas() {{
    canvas.width        = vid.clientWidth;
    canvas.height       = vid.clientHeight;
    canvas.style.width  = vid.clientWidth  + 'px';
    canvas.style.height = vid.clientHeight + 'px';
  }}

  function drawFrame() {{
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (showBoxes) {{
      const frameIdx  = String(Math.floor(vid.currentTime * FPS));
      const frameData = BOXES[frameIdx] || [];
      const scaleX    = canvas.width  / VID_W;
      const scaleY    = canvas.height / VID_H;

      ctx.strokeStyle = '#00ff44';
      ctx.lineWidth   = 2;
      ctx.fillStyle   = '#00ff44';
      ctx.font        = 'bold 13px sans-serif';

      for (const [x1, y1, x2, y2, tid] of frameData) {{
        const sx = x1 * scaleX, sy = y1 * scaleY;
        const sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.fillText('ID ' + tid, sx + 2, Math.max(sy - 4, 12));
      }}
    }}
    requestAnimationFrame(drawFrame);
  }}

  function toggleBoxes() {{
    showBoxes = !showBoxes;
    const btn = document.getElementById('toggleBtn');
    btn.textContent = 'Player Detection Boxes: ' + (showBoxes ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showBoxes);
  }}

  vid.addEventListener('loadedmetadata', () => {{ resizeCanvas(); drawFrame(); }});
  window.addEventListener('resize', resizeCanvas);
</script>
"""


def build_ball_html(video_url: str, frame_ball_boxes: dict, frame_gaussian: dict,
                    frame_w: int, frame_h: int, fps: float) -> str:
    ball_json     = json.dumps(frame_ball_boxes)
    gaussian_json = json.dumps(frame_gaussian)
    return f"""
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #000; }}

  #player-wrap {{
    position: relative;
    width: 100%;
    background: #000;
  }}
  #vid {{
    width: 100%;
    height: auto;
    display: block;
  }}
  #overlay {{
    position: absolute;
    top: 0; left: 0;
    pointer-events: none;
  }}
  #controls {{
    height: 44px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    background: #111;
  }}
  .toggleBtn {{
    padding: 5px 18px;
    font-size: 13px;
    font-weight: bold;
    border-radius: 6px;
    cursor: pointer;
  }}
  #ballBoxBtn {{
    border: 2px solid #ffff00;
    background: #ffff00;
    color: #fff;
  }}
  #ballBoxBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
  #gaussianBtn {{
    border: 2px solid #ff2222;
    background: #ff2222;
    color: #fff;
  }}
  #gaussianBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
</style>

<div id="controls">
  <button id="ballBoxBtn"  class="toggleBtn" onclick="toggleBallBox()">Ball Box: ON</button>
  <button id="gaussianBtn" class="toggleBtn" onclick="toggleGaussian()">Gaussian Smoothing: ON</button>
</div>
<div id="player-wrap">
  <video id="vid" controls>
    <source src="{video_url}" type="video/mp4">
  </video>
  <canvas id="overlay"></canvas>
</div>

<script>
  const BALL_BOXES = {ball_json};
  const GAUSSIAN   = {gaussian_json};
  const FPS   = {fps};
  const VID_W = {frame_w};
  const VID_H = {frame_h};
  let showBallBox  = true;
  let showGaussian = true;

  const vid    = document.getElementById('vid');
  const canvas = document.getElementById('overlay');
  const ctx    = canvas.getContext('2d');

  function resizeCanvas() {{
    canvas.width        = vid.clientWidth;
    canvas.height       = vid.clientHeight;
    canvas.style.width  = vid.clientWidth  + 'px';
    canvas.style.height = vid.clientHeight + 'px';
  }}

  function drawFrame() {{
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frameIdx = String(Math.floor(vid.currentTime * FPS));
    const scaleX   = canvas.width  / VID_W;
    const scaleY   = canvas.height / VID_H;

    if (showBallBox && BALL_BOXES[frameIdx]) {{
      const [x1, y1, x2, y2, conf] = BALL_BOXES[frameIdx];
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth   = 3;
      ctx.strokeRect(x1*scaleX, y1*scaleY, (x2-x1)*scaleX, (y2-y1)*scaleY);
      ctx.fillStyle = '#ffff00';
      ctx.font      = 'bold 13px sans-serif';
      ctx.fillText(Math.round(conf*100) + '%', x1*scaleX + 2, y1*scaleY - 4);
    }}

    if (showGaussian && GAUSSIAN[frameIdx]) {{
      const [px, py] = GAUSSIAN[frameIdx];
      ctx.strokeStyle = '#ff2222';
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.arc(px*scaleX, py*scaleY, 12, 0, 2*Math.PI);
      ctx.stroke();
    }}

    requestAnimationFrame(drawFrame);
  }}

  function toggleBallBox() {{
    showBallBox = !showBallBox;
    const btn = document.getElementById('ballBoxBtn');
    btn.textContent = 'Ball Box: ' + (showBallBox ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showBallBox);
  }}

  function toggleGaussian() {{
    showGaussian = !showGaussian;
    const btn = document.getElementById('gaussianBtn');
    btn.textContent = 'Gaussian Smoothing: ' + (showGaussian ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showGaussian);
  }}

  vid.addEventListener('loadedmetadata', () => {{ resizeCanvas(); drawFrame(); }});
  window.addEventListener('resize', resizeCanvas);
</script>
"""


def build_final_product_html(video_url: str, frame_boxes: dict,
                             sc_ball_boxes: dict, sc_pred: dict,
                             smoothed_x1s: list,
                             sx: float, sy: float, fps: float) -> str:
    player_json  = json.dumps(frame_boxes)
    ball_json    = json.dumps(sc_ball_boxes)
    pred_json    = json.dumps(sc_pred)
    crop_x1_json = json.dumps(smoothed_x1s)
    return f"""
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #000; display: flex; flex-direction: column; align-items: center; }}

  #controls {{
    width: 100%;
    max-width: 420px;
    min-height: 44px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    background: #111;
  }}
  .toggleBtn {{
    padding: 5px 14px;
    font-size: 13px;
    font-weight: bold;
    border-radius: 6px;
    cursor: pointer;
  }}
  #playerBtn {{
    border: 2px solid #00ff44;
    background: #00ff44;
    color: #000;
  }}
  #playerBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
  #ballBoxBtn {{
    border: 2px solid #ffff00;
    background: #ffff00;
    color: #000;
  }}
  #ballBoxBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
  #predBtn {{
    border: 2px solid #ff6400;
    background: #ff6400;
    color: #fff;
  }}
  #predBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
  #player-wrap {{
    position: relative;
    width: 100%;
    max-width: 420px;
    background: #000;
  }}
  #vid {{
    width: 100%;
    height: auto;
    display: block;
  }}
  #overlay {{
    position: absolute;
    top: 0; left: 0;
    pointer-events: none;
  }}
</style>

<div id="controls">
  <button id="playerBtn"  class="toggleBtn" onclick="togglePlayer()">Player Boxes: ON</button>
  <button id="ballBoxBtn" class="toggleBtn" onclick="toggleBall()">Ball Box: ON</button>
  <button id="predBtn"    class="toggleBtn" onclick="togglePred()">Tracker Prediction: ON</button>
</div>
<div id="player-wrap">
  <video id="vid" controls>
    <source src="{video_url}" type="video/mp4">
  </video>
  <canvas id="overlay"></canvas>
</div>

<script>
  const PLAYER_BOXES = {player_json};
  const BALL_BOXES   = {ball_json};
  const CROP_X1S     = {crop_x1_json};
  const PRED         = {pred_json};
  const SX  = {sx};
  const SY  = {sy};
  const FPS = {fps};
  const OUT_W = 540;
  const OUT_H = 960;
  let showPlayer = true;
  let showBall   = true;
  let showPred   = true;

  const vid    = document.getElementById('vid');
  const canvas = document.getElementById('overlay');
  const ctx    = canvas.getContext('2d');

  function resizeCanvas() {{
    canvas.width        = vid.clientWidth;
    canvas.height       = vid.clientHeight;
    canvas.style.width  = vid.clientWidth  + 'px';
    canvas.style.height = vid.clientHeight + 'px';
  }}

  function drawFrame() {{
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frameIdx = Math.floor(vid.currentTime * FPS);
    const scaleX   = canvas.width  / OUT_W;
    const scaleY   = canvas.height / OUT_H;
    const cropX1   = CROP_X1S[frameIdx] || 0;

    if (showPlayer) {{
      const frameData = PLAYER_BOXES[String(frameIdx)] || [];
      ctx.strokeStyle = '#00ff44';
      ctx.lineWidth   = 2;
      ctx.fillStyle   = '#00ff44';
      ctx.font        = 'bold 12px sans-serif';
      for (const [x1, y1, x2, y2, tid] of frameData) {{
        const cx1 = (x1 - cropX1) * SX * scaleX;
        const cy1 = y1 * SY * scaleY;
        const cw  = (x2 - x1) * SX * scaleX;
        const ch  = (y2 - y1) * SY * scaleY;
        if (cx1 + cw < 0 || cx1 > canvas.width) continue;
        ctx.strokeRect(cx1, cy1, cw, ch);
        ctx.fillText('ID ' + tid, cx1 + 2, Math.max(cy1 - 4, 12));
      }}
    }}

    if (showBall && BALL_BOXES[String(frameIdx)]) {{
      const [bx1, by1, bx2, by2, conf] = BALL_BOXES[String(frameIdx)];
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth   = 3;
      ctx.strokeRect(bx1 * scaleX, by1 * scaleY, (bx2 - bx1) * scaleX, (by2 - by1) * scaleY);
      ctx.fillStyle = '#ffff00';
      ctx.font      = 'bold 13px sans-serif';
      ctx.fillText(Math.round(conf * 100) + '%', bx1 * scaleX + 2, Math.max(14, by1 * scaleY - 4));
    }}

    if (showPred && PRED[String(frameIdx)]) {{
      const [px, py] = PRED[String(frameIdx)];
      ctx.strokeStyle = '#ff6400';
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.arc(px * scaleX, py * scaleY, 14, 0, 2 * Math.PI);
      ctx.stroke();
    }}

    requestAnimationFrame(drawFrame);
  }}

  function togglePlayer() {{
    showPlayer = !showPlayer;
    const btn = document.getElementById('playerBtn');
    btn.textContent = 'Player Boxes: ' + (showPlayer ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showPlayer);
  }}

  function toggleBall() {{
    showBall = !showBall;
    const btn = document.getElementById('ballBoxBtn');
    btn.textContent = 'Ball Box: ' + (showBall ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showBall);
  }}

  function togglePred() {{
    showPred = !showPred;
    const btn = document.getElementById('predBtn');
    btn.textContent = 'Tracker Prediction: ' + (showPred ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showPred);
  }}

  vid.addEventListener('loadedmetadata', () => {{ resizeCanvas(); drawFrame(); }});
  window.addEventListener('resize', resizeCanvas);
</script>
"""


def build_smart_crop_html(video_url: str, frame_ball_boxes: dict,
                           frame_pred: dict, fps: float) -> str:
    ball_json = json.dumps(frame_ball_boxes)
    pred_json = json.dumps(frame_pred)
    return f"""
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #000; display: flex; flex-direction: column; align-items: center; }}

  #controls {{
    width: 100%;
    max-width: 420px;
    height: 44px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    background: #111;
  }}
  .toggleBtn {{
    padding: 5px 14px;
    font-size: 13px;
    font-weight: bold;
    border-radius: 6px;
    cursor: pointer;
  }}
  #ballBoxBtn {{
    border: 2px solid #ffff00;
    background: #ffff00;
    color: #000;
  }}
  #ballBoxBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
  #predBtn {{
    border: 2px solid #ff6400;
    background: #ff6400;
    color: #fff;
  }}
  #predBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
  #player-wrap {{
    position: relative;
    width: 100%;
    max-width: 420px;
    background: #000;
  }}
  #vid {{
    width: 100%;
    height: auto;
    display: block;
  }}
  #overlay {{
    position: absolute;
    top: 0; left: 0;
    pointer-events: none;
  }}
</style>

<div id="controls">
  <button id="ballBoxBtn" class="toggleBtn" onclick="toggleBallBox()">Ball Box: ON</button>
  <button id="predBtn"    class="toggleBtn" onclick="togglePred()">Tracker Prediction: ON</button>
</div>
<div id="player-wrap">
  <video id="vid" controls>
    <source src="{video_url}" type="video/mp4">
  </video>
  <canvas id="overlay"></canvas>
</div>

<script>
  const BALL_BOXES = {ball_json};
  const PRED       = {pred_json};
  const FPS        = {fps};
  let showBallBox = true;
  let showPred    = true;

  const vid    = document.getElementById('vid');
  const canvas = document.getElementById('overlay');
  const ctx    = canvas.getContext('2d');

  function resizeCanvas() {{
    canvas.width        = vid.clientWidth;
    canvas.height       = vid.clientHeight;
    canvas.style.width  = vid.clientWidth  + 'px';
    canvas.style.height = vid.clientHeight + 'px';
  }}

  function drawFrame() {{
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frameIdx = String(Math.floor(vid.currentTime * FPS));
    const scaleX   = canvas.width  / 540;
    const scaleY   = canvas.height / 960;

    if (showBallBox && BALL_BOXES[frameIdx]) {{
      const [x1, y1, x2, y2, conf] = BALL_BOXES[frameIdx];
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth   = 3;
      ctx.strokeRect(x1*scaleX, y1*scaleY, (x2-x1)*scaleX, (y2-y1)*scaleY);
      ctx.fillStyle = '#ffff00';
      ctx.font      = 'bold 13px sans-serif';
      ctx.fillText(Math.round(conf*100) + '%', x1*scaleX + 2, Math.max(14, y1*scaleY - 4));
    }}

    if (showPred && PRED[frameIdx]) {{
      const [px, py] = PRED[frameIdx];
      ctx.strokeStyle = '#ff6400';
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.arc(px*scaleX, py*scaleY, 14, 0, 2*Math.PI);
      ctx.stroke();
    }}

    requestAnimationFrame(drawFrame);
  }}

  function toggleBallBox() {{
    showBallBox = !showBallBox;
    const btn = document.getElementById('ballBoxBtn');
    btn.textContent = 'Ball Box: ' + (showBallBox ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showBallBox);
  }}

  function togglePred() {{
    showPred = !showPred;
    const btn = document.getElementById('predBtn');
    btn.textContent = 'Tracker Prediction: ' + (showPred ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showPred);
  }}

  vid.addEventListener('loadedmetadata', () => {{ resizeCanvas(); drawFrame(); }});
  window.addEventListener('resize', resizeCanvas);
</script>
"""