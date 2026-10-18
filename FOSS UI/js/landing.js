/**
 * =========================================================
 * RESERVE*JUGGLER — EDITORIAL LANDING PAGE SCRIPT
 * Background Currency Stream + Live Portal Graph Simulation
 * =========================================================
 */

(function () {
  'use strict';

  let currencyEngine = null;
  let liveCashVal = 24810500;
  let simHour = 5920;
  let isAutoStreaming = false;
  let autoStreamInterval = null;

  // Portal Waveform Data
  const wavePointsCount = 32;
  let waveOffsets = new Array(wavePointsCount).fill(0).map((_, i) => Math.sin(i * 0.4) * 25);
  let targetWaveOffsets = [...waveOffsets];
  let pulseHead = 0;
  let shockActive = false;

  // Initialize background currency stream (Increased Density)
  function initCurrencyStream() {
    if (typeof CurrencyStreamEngine !== 'undefined') {
      currencyEngine = new CurrencyStreamEngine('currencyCanvas', {
        noteCount: 88,
        speed: 1.25,
        windAngle: 0.35,
        enableMouseWake: true
      });
    }
  }

  // Live Portal Graph Simulation Loop (60 FPS)
  function initPortalLiveGraph() {
    const canvas = document.getElementById('portalLiveGraphCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    function resizeGraph() {
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0) return;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }
    resizeGraph();
    window.addEventListener('resize', resizeGraph);

    let t = 0;

    function loop() {
      t += 0.035;
      pulseHead = (pulseHead + 0.3) % canvas.offsetWidth;

      // Animate wave offsets toward targets
      for (let i = 0; i < wavePointsCount; i++) {
        waveOffsets[i] += (targetWaveOffsets[i] - waveOffsets[i]) * 0.08;
      }

      const W = canvas.offsetWidth;
      const H = canvas.offsetHeight;
      ctx.clearRect(0, 0, W, H);

      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      const mainCyan = isLight ? '#0088A3' : '#00E5FF';
      const gridCol = isLight ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.05)';

      // Subtle Gridlines
      ctx.strokeStyle = gridCol;
      ctx.lineWidth = 1;
      for (let y = 20; y < H; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }

      // Live Oscillating Spline
      ctx.beginPath();
      ctx.strokeStyle = shockActive ? '#F43F5E' : mainCyan;
      ctx.lineWidth = 2.5;

      const pts = [];
      for (let i = 0; i < wavePointsCount; i++) {
        const x = (i / (wavePointsCount - 1)) * W;
        const dynamicSine = Math.sin(t + i * 0.35) * (shockActive ? 32 : 14);
        const y = H * 0.52 + waveOffsets[i] + dynamicSine;
        pts.push({ x, y });
      }

      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 0; i < pts.length - 1; i++) {
        const xc = (pts[i].x + pts[i + 1].x) / 2;
        const yc = (pts[i].y + pts[i + 1].y) / 2;
        ctx.quadraticCurveTo(pts[i].x, pts[i].y, xc, yc);
      }
      ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
      ctx.stroke();

      // Shaded Gradient Fill Under Curve
      ctx.lineTo(W, H);
      ctx.lineTo(0, H);
      ctx.closePath();
      const fillGrad = ctx.createLinearGradient(0, 0, 0, H);
      fillGrad.addColorStop(0, shockActive ? 'rgba(244, 63, 94, 0.22)' : (isLight ? 'rgba(0, 136, 163, 0.18)' : 'rgba(0, 229, 255, 0.22)'));
      fillGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = fillGrad;
      ctx.fill();

      // Traveling Laser Pulse Head Node
      const currentPointIdx = Math.min(pts.length - 1, Math.floor((pulseHead / W) * pts.length));
      if (pts[currentPointIdx]) {
        const p = pts[currentPointIdx];
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = shockActive ? '#F43F5E' : '#00E5FF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.stroke();
      }

      requestAnimationFrame(loop);
    }

    requestAnimationFrame(loop);
  }

  // Advance simulation step from landing page
  window.stepLandingSim = function (hours = 1) {
    simHour += hours;
    const delta = (Math.random() - 0.45) * 1200000;
    liveCashVal = Math.max(18000000, Math.min(34000000, liveCashVal + delta));

    // Update Counter with GSAP
    const el = document.getElementById('portalCashDisplay');
    if (el) {
      if (typeof gsap !== 'undefined') {
        gsap.to({ val: parseFloat(el.dataset.val || liveCashVal) }, {
          val: liveCashVal,
          duration: 0.75,
          ease: 'power2.out',
          onUpdate: function () {
            const v = Math.round(this.targets()[0].val);
            el.dataset.val = v;
            el.textContent = '$' + v.toLocaleString('en-US');
          }
        });
      } else {
        el.textContent = '$' + Math.round(liveCashVal).toLocaleString('en-US');
      }
    }

    const stepEl = document.getElementById('portalStepBadge');
    if (stepEl) stepEl.textContent = `H-${simHour}`;

    // Morph the wave targets
    for (let i = 0; i < wavePointsCount; i++) {
      targetWaveOffsets[i] = (Math.random() - 0.5) * 45;
    }
  };

  // Toggle Auto Stream on Landing Page
  window.toggleLandingAutoStream = function () {
    const btn = document.getElementById('landingAutoBtn');
    if (!btn) return;

    if (isAutoStreaming) {
      clearInterval(autoStreamInterval);
      isAutoStreaming = false;
      btn.innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-500"></span><span>Auto-Stream</span>';
    } else {
      isAutoStreaming = true;
      btn.innerHTML = '<span class="w-2 h-2 rounded-full bg-cyan-400 pulse-dot"></span><span>Streaming...</span>';
      autoStreamInterval = setInterval(() => {
        window.stepLandingSim(1);
      }, 1500);
    }
  };

  // Trigger SVB Shock Simulation
  window.triggerLandingShock = function () {
    shockActive = !shockActive;
    const btn = document.getElementById('landingShockBtn');
    const badge = document.getElementById('portalStatusBadge');

    if (shockActive) {
      if (btn) btn.classList.add('bg-rose-500/20', 'border-rose-500', 'text-rose-400');
      if (badge) {
        badge.textContent = '3.2x Run Shock Active';
        badge.className = 'pill-badge text-rose-400 border-rose-500 bg-rose-500/10 font-bold';
      }
      for (let i = 0; i < wavePointsCount; i++) {
        targetWaveOffsets[i] = (i / wavePointsCount) * 55 - 20;
      }
    } else {
      if (btn) btn.classList.remove('bg-rose-500/20', 'border-rose-500', 'text-rose-400');
      if (badge) {
        badge.textContent = 'Conformal VaR 99.5%';
        badge.className = 'pill-badge pill-badge-cyan font-bold';
      }
      for (let i = 0; i < wavePointsCount; i++) {
        targetWaveOffsets[i] = Math.sin(i * 0.4) * 20;
      }
    }
  };

  // Segmented Theme Controller (Default: Light Mode)
  window.setAppTheme = function (theme) {
    const html = document.documentElement;
    html.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('rj_theme', theme);
    } catch (e) {}

    const btnLight = document.getElementById('themeBtnLight');
    const btnDark = document.getElementById('themeBtnDark');
    if (btnLight && btnDark) {
      if (theme === 'dark') {
        btnLight.classList.remove('active');
        btnDark.classList.add('active');
      } else {
        btnLight.classList.add('active');
        btnDark.classList.remove('active');
      }
    }
  };

  // Toggle fallback for shortcut or generic toggle
  window.toggleLandingTheme = function () {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'light' ? 'dark' : 'light';
    window.setAppTheme(next);
  };
  window.toggleTheme = window.toggleLandingTheme;

  // Initialize theme from storage, strictly defaulting to 'light'
  (function initTheme() {
    let saved = 'light';
    try {
      saved = localStorage.getItem('rj_theme') || 'light';
    } catch (e) {
      saved = 'light';
    }
    window.setAppTheme(saved);
  })();

  // 3D Parallax Tilt for the Circular Portal
  function initPortalParallax() {
    const portal = document.querySelector('.hero-circular-portal');
    if (!portal) return;

    window.addEventListener('mousemove', (e) => {
      const cx = window.innerWidth * 0.7;
      const cy = window.innerHeight * 0.5;
      const dx = (e.clientX - cx) / (window.innerWidth * 0.5);
      const dy = (e.clientY - cy) / (window.innerHeight * 0.5);

      portal.style.transform = `perspective(1000px) rotateY(${dx * 9}deg) rotateX(${-dy * 9}deg)`;
    });
  }

  // DOM Loaded
  window.addEventListener('DOMContentLoaded', () => {
    initCurrencyStream();
    initPortalLiveGraph();
    initPortalParallax();

    // GSAP Entrance
    if (typeof gsap !== 'undefined') {
      gsap.from('.editorial-reveal', {
        y: 28,
        opacity: 0,
        stagger: 0.14,
        duration: 0.85,
        ease: 'power3.out'
      });
      gsap.from('.hero-circular-portal', {
        scale: 0.88,
        opacity: 0,
        duration: 1.1,
        ease: 'back.out(1.4)'
      });
    }
  });
})();
