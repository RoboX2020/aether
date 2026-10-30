// ============================================================================
// Aether - an instrument you play with your hands in mid-air.
//
// HOW TO PLAY: open the page, click "begin", allow the webcam.
//   RIGHT hand: its height/position picks the musical note; pinch thumb +
//               index to make the note sound.
//   LEFT hand:  height controls loudness and brightness; pinch to toggle a
//               big echo (reverb) on and off.
//
// THE THREE FILES:
//   index.html  the page skeleton (video, canvas, start screen)
//   style.css   the look (dark, glowy, neon particles)
//   app.js      THIS FILE: all the behavior
//
// HOW APP.JS WORKS, BIG PICTURE:
//   1. MediaPipe (Google's hand-tracking AI, loaded from a CDN) watches the
//      webcam and reports 21 landmarks per hand, several times a second.
//   2. The Web Audio API (built into the browser) synthesizes the sound
//      from raw oscillators - there are NO audio files in this project.
//   3. A canvas draws the glowing skeletons and particle trails.
//   4. loop() runs once per screen frame and ties it all together.
//
// WHEN DEBUGGING (reasoned from the code):
//   - Nothing happens on click: the browser blocks audio until a user
//     gesture; the begin button is that gesture. Check the console.
//   - No hands found: needs decent light; the AI model downloads from a
//     Google CDN on first load, so first "begin" needs internet.
//   - Left/right feel swapped: the camera image is mirrored on purpose;
//     see classifyHands() for how hands are told apart.
// ============================================================================

// Google's hand-tracking AI, loaded straight from a CDN (no npm install).
import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// The page pieces this script talks to (all defined in index.html).
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const splash = document.getElementById("splash");
const startBtn = document.getElementById("start");
const statusEl = document.getElementById("status");
const noteEl = document.getElementById("note");

let handLandmarker = null;
let audio = null;
let lastVideoTime = -1;
let particles = [];
let reverbOn = false;
let reverbLatch = false;
let lastNoteName = "—";

// Keeps the drawing canvas matching the window size, sharp on retina
// screens (dpr = device pixels per CSS pixel, capped at 2 for performance).
function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  canvas.style.width = window.innerWidth + "px";
  canvas.style.height = window.innerHeight + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// Builds the "echo fingerprint" for the reverb effect: a few seconds of
// noise that fades out. The convolver later stamps this fingerprint onto
// the sound, which is what makes it feel like playing in a big hall.
function makeImpulseResponse(actx, duration, decay) {
  const sampleRate = actx.sampleRate;
  const length = Math.floor(sampleRate * duration);
  const impulse = actx.createBuffer(2, length, sampleRate);
  for (let c = 0; c < 2; c++) {
    const data = impulse.getChannelData(c);
    for (let i = 0; i < length; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
    }
  }
  return impulse;
}

// Builds the whole synthesizer and returns handles to its knobs.
// Picture a small studio: two detuned "sawtooth" oscillators (the main
// buzzy tone, slightly detuned against each other for warmth) plus a "sine"
// sub-oscillator one octave down (the body). Those run through a lowpass
// filter (the tone knob), a volume knob (noteGain), a left-right panner,
// then split into a dry path and a reverb ("wet") path that rejoin at the
// master volume, through a loudness meter (analyser), out to the speakers.
function makeAudio() {
  const actx = new (window.AudioContext || window.webkitAudioContext)();

  const osc1 = actx.createOscillator();
  osc1.type = "sawtooth";
  osc1.detune.value = -7;
  const osc2 = actx.createOscillator();
  osc2.type = "sawtooth";
  osc2.detune.value = 7;
  const sub = actx.createOscillator();
  sub.type = "sine";

  const oscMix = actx.createGain();
  oscMix.gain.value = 0.4;
  const subMix = actx.createGain();
  subMix.gain.value = 0.5;

  const filter = actx.createBiquadFilter();
  filter.type = "lowpass";
  filter.frequency.value = 1200;
  filter.Q.value = 4;

  const noteGain = actx.createGain();
  noteGain.gain.value = 0;

  const panner = actx.createStereoPanner();

  const dryGain = actx.createGain();
  dryGain.gain.value = 1;
  const wetGain = actx.createGain();
  wetGain.gain.value = 0;

  const convolver = actx.createConvolver();
  convolver.buffer = makeImpulseResponse(actx, 3.2, 2.4);

  const masterGain = actx.createGain();
  masterGain.gain.value = 0.5;

  const analyser = actx.createAnalyser();
  analyser.fftSize = 512;
  const ampData = new Uint8Array(analyser.frequencyBinCount);

  // Routing
  osc1.connect(oscMix);
  osc2.connect(oscMix);
  sub.connect(subMix);
  oscMix.connect(filter);
  subMix.connect(filter);
  filter.connect(noteGain);
  noteGain.connect(panner);
  panner.connect(dryGain);
  panner.connect(convolver);
  convolver.connect(wetGain);
  dryGain.connect(masterGain);
  wetGain.connect(masterGain);
  masterGain.connect(analyser);
  analyser.connect(actx.destination);

  osc1.start();
  osc2.start();
  sub.start();

  return {
    ctx: actx,
    osc1,
    osc2,
    sub,
    filter,
    noteGain,
    panner,
    dryGain,
    wetGain,
    masterGain,
    analyser,
    ampData,
  };
}

// Glides all three oscillators to a new pitch. setTargetAtTime with a small
// time constant (0.03s) means "slide there quickly but smoothly" - no harsh
// pitch jumps.
function setPitch(freq) {
  const t = audio.ctx.currentTime;
  audio.osc1.frequency.setTargetAtTime(freq, t, 0.03);
  audio.osc2.frequency.setTargetAtTime(freq, t, 0.03);
  audio.sub.frequency.setTargetAtTime(freq * 0.5, t, 0.03);
}

// C major pentatonic gives a friendly, always-in-tune sound.
const PENTATONIC = [0, 2, 4, 7, 9];

// Snaps any frequency to the nearest note of the pentatonic scale, so
// wherever your hand lands, it sounds musical - never between the keys.
function snapPentatonic(freq) {
  const A4 = 440;
  const semi = 12 * Math.log2(freq / A4);
  const oct = Math.floor(semi / 12);
  const within = ((semi % 12) + 12) % 12;
  let best = PENTATONIC[0];
  let bestDiff = Infinity;
  for (const s of PENTATONIC) {
    const d = Math.abs(within - s);
    if (d < bestDiff) {
      bestDiff = d;
      best = s;
    }
  }
  const snappedSemi = oct * 12 + best;
  return A4 * Math.pow(2, snappedSemi / 12);
}

const NOTE_NAMES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"];
// Turns a frequency into a display name like "A4" or "C#5" for the HUD.
function freqToName(freq) {
  const A4 = 440;
  const semi = Math.round(12 * Math.log2(freq / A4));
  const octave = 4 + Math.floor((semi + 9) / 12);
  const idx = ((semi % 12) + 12) % 12;
  return `${NOTE_NAMES[idx]}${octave}`;
}

// Downloads and starts the hand-tracking AI (runs on the GPU if available,
// tracks up to 2 hands, 50% confidence minimums).
async function initHands() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

// Asks for the webcam (video only, no mic) and starts playing it into the
// hidden <video> element.
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((r) => (video.onloadedmetadata = r));
  await video.play();
}

// Plain distance between two hand landmarks.
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Split detected hands by raw camera X so the assignment is stable.
// Mirroring on display flips left/right: user's right hand has LOW raw x.
function classifyHands(landmarks) {
  if (!landmarks || landmarks.length === 0) return { right: null, left: null };
  const sorted = [...landmarks].sort((a, b) => a[0].x - b[0].x);
  if (sorted.length === 1) {
    return sorted[0][0].x < 0.5 ? { right: sorted[0], left: null } : { right: null, left: sorted[0] };
  }
  return { right: sorted[0], left: sorted[1] };
}

// How pinched is this hand, as a ratio? Thumb tip (4) to index tip (8)
// distance, divided by hand size (wrist 0 to middle knuckle 9) so the same
// pinch reads the same whether your hand is near or far from the camera.
function pinchAmount(hand) {
  const tip = hand[8];
  const thumb = hand[4];
  const scale = dist(hand[0], hand[9]) || 0.001;
  return dist(tip, thumb) / scale; // ~0.2 = pinched, ~1.0 = open
}

// Turns hand positions into sound, every frame.
//   right hand: left-right position = pitch, also pans the sound to match;
//               pinch = note on, release = note off (with a gentle fade)
//   left hand:  height = master volume and filter brightness;
//               pinch toggles the reverb (a "latch" makes one pinch = one
//               toggle, not machine-gun toggling while held)
function updateAudio(rightHand, leftHand) {
  if (!audio) return;
  const t = audio.ctx.currentTime;

  if (rightHand) {
    const tip = rightHand[8];
    const mx = 1 - tip.x; // mirror
    const my = tip.y;

    const minF = 130;
    const maxF = 880;
    const freq = minF * Math.pow(maxF / minF, mx);
    const snapped = snapPentatonic(freq);
    setPitch(snapped);
    lastNoteName = freqToName(snapped);

    audio.panner.pan.setTargetAtTime((mx - 0.5) * 1.6, t, 0.04);

    const pinched = pinchAmount(rightHand) < 0.35;
    audio.noteGain.gain.setTargetAtTime(pinched ? 0.55 : 0.0, t, pinched ? 0.02 : 0.12);
  } else {
    audio.noteGain.gain.setTargetAtTime(0, t, 0.15);
  }

  if (leftHand) {
    const tip = leftHand[8];
    const my = 1 - tip.y; // hand high = bright/loud

    audio.masterGain.gain.setTargetAtTime(0.2 + my * 0.6, t, 0.05);
    const cutoff = 220 * Math.pow(28, my);
    audio.filter.frequency.setTargetAtTime(cutoff, t, 0.05);

    const pinched = pinchAmount(leftHand) < 0.35;
    if (pinched && !reverbLatch) {
      reverbLatch = true;
      reverbOn = !reverbOn;
      audio.wetGain.gain.setTargetAtTime(reverbOn ? 0.7 : 0.0, t, 0.25);
      audio.dryGain.gain.setTargetAtTime(reverbOn ? 0.55 : 1.0, t, 0.25);
    } else if (!pinched) {
      reverbLatch = false;
    }
  }
}

// Spawns a few glowing particles at each fingertip (extra from the index
// finger, the "playing" one). hue = color: 340 magenta (right), 200 blue
// (left).
function emitFromHand(hand, hue) {
  if (!hand) return;
  const w = window.innerWidth;
  const h = window.innerHeight;
  const tips = [4, 8, 12, 16, 20];
  for (const t of tips) {
    const x = (1 - hand[t].x) * w;
    const y = hand[t].y * h;
    const count = t === 8 ? 3 : 1;
    for (let i = 0; i < count; i++) {
      particles.push({
        x,
        y,
        vx: (Math.random() - 0.5) * 1.4,
        vy: (Math.random() - 0.5) * 1.4 - 0.4,
        life: 1.0,
        size: 1.5 + Math.random() * 3.5,
        hue: hue + (Math.random() - 0.5) * 20,
      });
    }
  }
}

// Draws the faint hand skeleton lines connecting the landmarks finger by
// finger (the arrays are the bone chains of MediaPipe's 21-point hand map).
function drawHandSkeleton(hand, hue) {
  if (!hand) return;
  const w = window.innerWidth;
  const h = window.innerHeight;
  const fingers = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [5, 9, 13, 17],
  ];
  ctx.strokeStyle = `hsla(${hue}, 80%, 75%, 0.35)`;
  ctx.lineWidth = 1.2;
  for (const chain of fingers) {
    ctx.beginPath();
    for (let i = 0; i < chain.length; i++) {
      const lm = hand[chain[i]];
      const x = (1 - lm.x) * w;
      const y = lm.y * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

// Moves, ages and draws all particles. They drift, fall slowly, fade out,
// and swell with the current loudness (amp). "lighter" blending makes
// overlapping particles add their light - that is the neon glow.
function drawParticles(amp) {
  ctx.globalCompositeOperation = "lighter";
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.012;
    p.life -= 0.011;
    if (p.life <= 0) {
      particles.splice(i, 1);
      continue;
    }
    const r = p.size * (1 + amp * 2.5);
    ctx.fillStyle = `hsla(${p.hue}, 90%, ${55 + amp * 25}%, ${p.life * 0.9})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";
}

// The heartbeat: runs once per screen frame via requestAnimationFrame.
// Each frame: fade the canvas a little (leaving motion trails), read the
// newest camera frame through the hand AI, turn hands into sound and
// visuals, then draw particles sized by how loud the synth currently is.
function loop() {
  // Fade with a translucent overlay to leave trails.
  ctx.fillStyle = "rgba(5, 3, 20, 0.22)";
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  if (handLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(video, performance.now());
    const { right, left } = classifyHands(result.landmarks);

    updateAudio(right, left);
    drawHandSkeleton(right, 340);
    drawHandSkeleton(left, 200);
    emitFromHand(right, 340);
    emitFromHand(left, 200);
  }

  let amp = 0;
  if (audio) {
    audio.analyser.getByteFrequencyData(audio.ampData);
    let sum = 0;
    for (let i = 0; i < audio.ampData.length; i++) sum += audio.ampData[i];
    amp = sum / audio.ampData.length / 255;
  }

  drawParticles(amp);

  noteEl.textContent = audio && audio.noteGain.gain.value > 0.05 ? lastNoteName : "—";

  requestAnimationFrame(loop);
}

// The "begin" button: ask for the camera, load the hand AI, build the
// synth, fade out the start screen, and start the loop. Browsers require a
// click like this before audio or camera are allowed to start.
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "loading…";
  try {
    statusEl.textContent = "asking for camera…";
    await initCamera();
    statusEl.textContent = "loading hand model…";
    await initHands();
    audio = makeAudio();
    if (audio.ctx.state === "suspended") await audio.ctx.resume();
    statusEl.textContent = "ready — pinch right thumb + index to play";
    splash.classList.add("hidden");
    setTimeout(() => (splash.style.display = "none"), 800);
    loop();
  } catch (e) {
    console.error(e);
    statusEl.textContent = `error: ${e.message}`;
    startBtn.disabled = false;
    startBtn.textContent = "try again";
  }
});
