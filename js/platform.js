/**
 * =========================================================
 * RESERVE*JUGGLER PLATFORM CONTROLLER — PLATFORM.JS
 * Canvas Graph Rendering, Tab Switching, and Hotkey HUD
 * =========================================================
 */

(function () {
  'use strict';

  const sim = new TreasurySimulationEngine();
  window.simEngine = sim;

  // View Switching
  window.switchView = function (viewId) {
    document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
    document.querySelectorAll('#navList .nav-tab-pill').forEach(btn => {
      btn.classList.remove('active');
      if (btn.getAttribute('data-nav') === viewId) btn.classList.add('active');
    });

    const activeSec = document.getElementById(`view-${viewId}`);
    if (activeSec) {
      activeSec.classList.add('active');
      runGsapEntrance(`#view-${viewId}`);
    }

    if (viewId === 'overview' || viewId === 'forecasts') {
      setTimeout(renderLiquidityCanvas, 40);
    } else if (viewId === 'portfolio') {
      setTimeout(renderPortfolioCanvases, 40);
    } else if (viewId === 'stress') {
      setTimeout(renderStressCanvas, 40);
    }
  };

  function runGsapEntrance(selector) {
    if (typeof gsap === 'undefined') return;
    gsap.fromTo(`${selector} .curved-card`,
      { y: 16, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.36, stagger: 0.04, ease: 'power3.out', clearProps: 'transform' }
    );
  }

  // Advancing steps
  window.advanceSimulationStep = function (hours = 1) {
    sim.advanceStep(hours);
    showToast(`Simulation advanced +${hours}h with dynamic spline morph.`);
  };

  // Auto-Play Stream
  window.toggleAutoPlay = function () {
    const btn = document.getElementById('autoPlayBtn');
    const dot = document.getElementById('autoPlayDot');
    const lbl = document.getElementById('autoPlayLabel');

    if (sim.autoPlayInterval) {
      clearInterval(sim.autoPlayInterval);
      sim.autoPlayInterval = null;
      dot.className = 'w-2 h-2 rounded-full bg-slate-500';
      lbl.textContent = 'Auto-Stream';
      showToast('Simulation stream paused.');
    } else {
      sim.autoPlayInterval = setInterval(() => {
        sim.advanceStep(1);
      }, 1400);
      dot.className = 'w-2 h-2 rounded-full bg-cyan-400 pulse-dot';
      lbl.textContent = 'Streaming...';
      showToast('Live autonomous stream active.');
    }
  };

  // Set Chart Horizon (24h, 48h, 72h)
  window.setChartHorizon = function (hours) {
    sim.currentHorizon = hours;
    document.querySelectorAll('.horizon-btn').forEach(btn => {
      btn.classList.toggle('active', parseInt(btn.dataset.horizon, 10) === hours);
    });
    sim.animateChartToNewData();
  };

  // Status Cipher flip
  window.triggerStatusCycle = function () {
    sim.triggerStatusCycle();
  };

  // Toast Notification
  window.showToast = function (msg) {
    const box = document.getElementById('toastBox');
    const text = document.getElementById('toastMsg');
    if (!box || !text) return;

    text.textContent = msg;
    box.classList.remove('hidden');

    if (typeof gsap !== 'undefined') {
      gsap.fromTo(box, { y: 12, opacity: 0 }, { y: 0, opacity: 1, duration: 0.22, ease: 'power3.out' });
    }

    clearTimeout(window._toastTimeout);
    window._toastTimeout = setTimeout(() => {
      box.classList.add('hidden');
    }, 3200);
  };

  // Card Selection Feedback
  window.selectCurvedCard = function (cardEl, message) {
    document.querySelectorAll('.curved-card').forEach(c => c.classList.remove('is-selected'));
    cardEl.classList.add('is-selected');
    if (typeof gsap !== 'undefined') {
      gsap.fromTo(cardEl, { scale: 0.985 }, { scale: 1, duration: 0.24, ease: 'back.out(2)' });
    }
    if (message) showToast(message);
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

    // Legacy button fallback if present
    const lbl = document.getElementById('themeLabel');
    const ico = document.getElementById('themeIcon');
    if (lbl) lbl.textContent = theme === 'dark' ? 'Light' : 'Dark';
    if (ico) ico.innerHTML = theme === 'dark' 
      ? '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4" stroke-width="2"></circle><path stroke-linecap="round" stroke-width="2" d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32l1.41 1.41M2 12h2m16 0h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"></path></svg>'
      : '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path></svg>';

    renderLiquidityCanvas();
    renderPortfolioCanvases();
    renderStressCanvas();
  };

  // Toggle Theme (Kept for shortcut 'T')
  window.toggleTheme = function () {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme') || 'light';
    const next = current === 'light' ? 'dark' : 'light';
    window.setAppTheme(next);
    showToast(`Switched to ${next.toUpperCase()} theme.`);
  };

  // Main Liquidity Spline Canvas
  window.renderLiquidityCanvas = function () {
    const canvas = document.getElementById('liquidityCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const styles = getComputedStyle(document.documentElement);
    const cyan = styles.getPropertyValue('--cyan-primary').trim() || '#00E5FF';
    const textMuted = styles.getPropertyValue('--text-muted').trim() || '#64748B';
    const gridCol = styles.getPropertyValue('--canvas-grid').trim() || 'rgba(255, 255, 255, 0.05)';

    const visiblePoints = sim.seriesData.filter(d => d.stepOffset <= sim.currentHorizon);
    const padL = 54, padR = 20, padT = 20, padB = 30;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const minVal = 14000000, maxVal = 38000000;

    const toX = (idx) => padL + (idx / (visiblePoints.length - 1)) * plotW;
    const toY = (val) => padT + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;

    // Grid lines
    ctx.strokeStyle = gridCol;
    ctx.lineWidth = 1;
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.fillStyle = textMuted;

    for (let v = 15000000; v <= 35000000; v += 5000000) {
      const y = toY(v);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
      ctx.fillText(`$${(v / 1000000).toFixed(0)}M`, 10, y + 3);
    }

    // VaR 99.5% Uncertainty Ribbon
    ctx.beginPath();
    visiblePoints.forEach((p, i) => {
      const x = toX(i);
      const y = toY(p.upper);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    for (let i = visiblePoints.length - 1; i >= 0; i--) {
      const x = toX(i);
      const y = toY(visiblePoints[i].lower);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = styles.getPropertyValue('--cyan-glow').trim() || 'rgba(0, 229, 255, 0.12)';
    ctx.fill();

    // Traveling Live Pulse Animation Loop State
    if (!window._graphAnimTime) window._graphAnimTime = 0;
    window._graphAnimTime += 0.04;
    const travelingPulseX = (window._graphAnimTime * 45) % (plotW);

    // Spline Cash Curve (interpolated via GSAP values + live subtle breath)
    const curvePoints = sim.currentPlotPoints.map((val, i) => {
      const breath = Math.sin(window._graphAnimTime + i * 0.45) * 220000;
      return {
        x: toX(i),
        y: toY(val + breath)
      };
    });

    if (curvePoints.length > 1) {
      // Glow under curve
      ctx.beginPath();
      ctx.strokeStyle = cyan;
      ctx.lineWidth = 2.5;

      ctx.moveTo(curvePoints[0].x, curvePoints[0].y);
      for (let i = 0; i < curvePoints.length - 1; i++) {
        const xc = (curvePoints[i].x + curvePoints[i + 1].x) / 2;
        const yc = (curvePoints[i].y + curvePoints[i + 1].y) / 2;
        ctx.quadraticCurveTo(curvePoints[i].x, curvePoints[i].y, xc, yc);
      }
      ctx.lineTo(curvePoints[curvePoints.length - 1].x, curvePoints[curvePoints.length - 1].y);
      ctx.stroke();

      // Shaded area under curve
      ctx.lineTo(toX(curvePoints.length - 1), padT + plotH);
      ctx.lineTo(padL, padT + plotH);
      ctx.closePath();
      const areaGrad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
      areaGrad.addColorStop(0, styles.getPropertyValue('--cyan-glow').trim() || 'rgba(0, 229, 255, 0.18)');
      areaGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = areaGrad;
      ctx.fill();

      // Continuous Traveling Laser Pulse Node on Curve
      const currentPointIdx = Math.min(curvePoints.length - 1, Math.floor((travelingPulseX / plotW) * curvePoints.length));
      if (curvePoints[currentPointIdx]) {
        const pt = curvePoints[currentPointIdx];
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = cyan;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 8 + Math.sin(window._graphAnimTime * 3) * 2, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Current Step Node
      const nowIdx = visiblePoints.findIndex(p => p.stepOffset === 0);
      if (nowIdx !== -1 && curvePoints[nowIdx]) {
        const pt = curvePoints[nowIdx];
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = cyan;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 10, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = cyan;
        ctx.font = 'bold 9px "IBM Plex Mono", monospace';
        ctx.fillText('17:00 Sweep', pt.x - 28, pt.y - 14);
      }
    }

    // Keep continuous 60 FPS animation active if canvas is visible
    if (!window._liquidityRafRunning) {
      window._liquidityRafRunning = true;
      function animateLiquidityCanvasLoop() {
        window.renderLiquidityCanvas();
        requestAnimationFrame(animateLiquidityCanvasLoop);
      }
      requestAnimationFrame(animateLiquidityCanvasLoop);
    }
  };

  // 4-Tier Portfolio Breakdown Canvases
  window.renderPortfolioCanvases = function () {
    const c = document.getElementById('portfolioWaveCanvas');
    if (!c) return;
    const ctx = c.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = c.getBoundingClientRect();
    if (rect.width === 0) return;

    c.width = rect.width * dpr;
    c.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const styles = getComputedStyle(document.documentElement);
    const cyan = styles.getPropertyValue('--cyan-primary').trim() || '#00E5FF';

    ctx.strokeStyle = cyan;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = 0; x < W; x += 4) {
      const y = H * 0.5 + Math.sin(x * 0.04) * (H * 0.35);
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  // SVB Stress Lab Canvas
  window.renderStressCanvas = function () {
    const c = document.getElementById('stressCanvas');
    if (!c) return;
    const ctx = c.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = c.getBoundingClientRect();
    if (rect.width === 0) return;

    c.width = rect.width * dpr;
    c.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const surge = parseFloat(document.getElementById('stressSurgeSlider')?.value || 1.0);
    const styles = getComputedStyle(document.documentElement);
    const critical = styles.getPropertyValue('--rose-accent').trim() || '#F43F5E';
    const cyan = styles.getPropertyValue('--cyan-primary').trim() || '#00E5FF';

    ctx.strokeStyle = surge > 2.0 ? critical : cyan;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let x = 0; x < W; x += 6) {
      const decay = Math.exp(-x * 0.005 * surge);
      const y = 30 + (H - 50) * (1 - decay);
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  window.updateStressLab = function () {
    const sEl = document.getElementById('stressSurgeSlider');
    const yEl = document.getElementById('stressYieldSlider');
    if (!sEl || !yEl) return;

    const surge = parseFloat(sEl.value);
    const yShock = parseInt(yEl.value, 10);
    const surgeVal = document.getElementById('stressSurgeVal');
    const yieldVal = document.getElementById('stressYieldVal');
    const minBufEl = document.getElementById('stressMinBuffer');

    if (surgeVal) surgeVal.textContent = `${surge.toFixed(1)}x`;
    if (yieldVal) yieldVal.textContent = `+${yShock} bps`;

    const minBuf = Math.max(0.5, 18.2 - (surge - 1.0) * 4.8);
    if (minBufEl) minBufEl.textContent = `$${minBuf.toFixed(2)}M`;

    renderStressCanvas();
  };

  window.applyStressPreset = function (s, y) {
    const sEl = document.getElementById('stressSurgeSlider');
    const yEl = document.getElementById('stressYieldSlider');
    if (sEl && yEl) {
      sEl.value = s;
      yEl.value = y;
      updateStressLab();
      showToast(`Stress preset applied: ${s}x surge.`);
    }
  };

  // Simulation Theatre & Demo Flow Integration
  let simTheatreAnimId = null;
  window.renderSimTheatreCanvas = function () {
    const c = document.getElementById('simTheatreCanvas');
    if (!c) return;
    const ctx = c.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = c.getBoundingClientRect();
    if (rect.width === 0) return;

    c.width = rect.width * dpr;
    c.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';

    // 4 Vault Tanks
    const tanks = [
      { name: 'Tier 0: Cash', sub: 'Operational Checking', val: '$24.81M', pct: 0.68, color: '#00E5FF', status: '100% LIQUID' },
      { name: 'Tier 1: Repo', sub: 'Overnight Facility', val: '$18.20M', pct: 0.52, color: '#10B981', status: 'LOCKED (4.30% APY)' },
      { name: 'Tier 2: Sovereign', sub: '2Y Treasury Debt', val: '$65.10M', pct: 0.82, color: '#F59E0B', status: 'YIELD (4.80% APY)' },
      { name: 'Tier 3: Equities', sub: 'Growth Index', val: '$51.90M', pct: 0.88, color: isLight ? '#0088A3' : '#00E5FF', status: '100% QUARANTINED' }
    ];

    const tWidth = (W - 90) / 4;
    const tHeight = H * 0.56;
    const tTop = H * 0.24;

    tanks.forEach((t, i) => {
      const tx = 20 + i * (tWidth + 16);

      // Tank Container
      ctx.fillStyle = isLight ? '#FFFFFF' : '#0F172A';
      ctx.strokeStyle = t.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(tx, tTop, tWidth, tHeight, 12);
      ctx.fill();
      ctx.stroke();

      // Fluid Reservoir
      const fluidH = tHeight * t.pct;
      const fluidY = tTop + tHeight - fluidH;
      ctx.fillStyle = t.color + (isLight ? '28' : '33');
      ctx.beginPath();
      ctx.roundRect(tx + 2, fluidY, tWidth - 4, fluidH - 2, [0, 0, 10, 10]);
      ctx.fill();

      // Fluid Wave Crest
      ctx.strokeStyle = t.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const time = performance.now() * 0.003;
      for (let x = tx; x <= tx + tWidth; x += 4) {
        const wy = fluidY + Math.sin(x * 0.06 + time + i * 1.5) * 4;
        if (x === tx) ctx.moveTo(x, wy); else ctx.lineTo(x, wy);
      }
      ctx.stroke();

      // Labels inside Tank
      ctx.fillStyle = isLight ? '#0A0F1D' : '#FFFFFF';
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(t.name, tx + tWidth / 2, tTop + 24);

      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.fillStyle = isLight ? '#64748B' : '#94A3B8';
      ctx.fillText(t.sub, tx + tWidth / 2, tTop + 40);

      ctx.font = 'bold 15px "IBM Plex Mono", monospace';
      ctx.fillStyle = t.color;
      ctx.fillText(t.val, tx + tWidth / 2, tTop + tHeight / 2 + 6);

      ctx.font = 'bold 9px "IBM Plex Mono", monospace';
      ctx.fillStyle = isLight ? '#475569' : '#CBD5E1';
      ctx.fillText(t.status, tx + tWidth / 2, tTop + tHeight - 16);

      // Connecting Conduits / Particles between tanks
      if (i < tanks.length - 1) {
        const cx = tx + tWidth;
        const cy = tTop + tHeight / 2;
        ctx.strokeStyle = isLight ? 'rgba(0, 136, 163, 0.35)' : 'rgba(0, 229, 255, 0.4)';
        ctx.lineWidth = 3;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + 16, cy);
        ctx.stroke();
        ctx.setLineDash([]);

        // Animated moving capital particle
        const pPhase = (performance.now() * 0.004 + i * 1.2) % 1;
        const px = cx + pPhase * 16;
        ctx.fillStyle = '#00E5FF';
        ctx.beginPath();
        ctx.arc(px, cy, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  };

  window.openSimulationModal = function (type = 'night_sweep') {
    const modal = document.getElementById('simTheatreOverlay');
    if (!modal) return;
    modal.classList.remove('hidden');
    modal.classList.add('flex');

    if (simTheatreAnimId) cancelAnimationFrame(simTheatreAnimId);
    function loop() {
      const m = document.getElementById('simTheatreOverlay');
      if (m && !m.classList.contains('hidden')) {
        renderSimTheatreCanvas();
        simTheatreAnimId = requestAnimationFrame(loop);
      }
    }
    loop();
  };

  window.closeSimulationModal = function () {
    const modal = document.getElementById('simTheatreOverlay');
    if (!modal) return;
    modal.classList.add('hidden');
    modal.classList.remove('flex');
    if (simTheatreAnimId) cancelAnimationFrame(simTheatreAnimId);
    simTheatreAnimId = null;
  };

  window.commitSimulationResult = function () {
    closeSimulationModal();
    showToast('Simulation state committed to live treasury ledger.');
  };

  // Next Best Action Arbitrage Approval
  window.approveNBA = function () {
    const btn = document.getElementById('approveNbaBtn');
    const badge = document.getElementById('nbaStatusBadge');
    if (btn) btn.textContent = '✓ Approved ($49,726 Saved)';
    if (badge) {
      badge.textContent = 'Staged // 08:00 Repay';
      badge.className = 'pill-badge pill-badge-emerald font-bold';
    }
    showToast('2-Hour Bridge facility approved.');
  };

  // Shortcuts Modal
  window.openShortcutsModal = function () {
    const m = document.getElementById('shortcutsModal');
    if (m) { m.classList.remove('hidden'); m.classList.add('flex'); }
  };
  window.closeShortcutsModal = function () {
    const m = document.getElementById('shortcutsModal');
    if (m) { m.classList.add('hidden'); m.classList.remove('flex'); }
  };

  // Mobile Nav
  window.toggleMobileNav = function () {
    document.getElementById('sidebarNav')?.classList.toggle('-translate-x-full');
  };

  // Global Keyboard Navigation
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'Escape') {
      closeSimulationModal();
      closeShortcutsModal();
    } else if (e.key === '?') {
      openShortcutsModal();
    } else if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault();
      advanceSimulationStep(1);
    } else if (e.key === 't' || e.key === 'T') {
      toggleTheme();
    } else if (e.key === 's' || e.key === 'S') {
      openSimulationModal('night_sweep');
    } else if (e.key >= '1' && e.key <= '7') {
      const viewKeys = ['overview', 'forecasts', 'portfolio', 'nba', 'stress', 'audit', 'settings'];
      switchView(viewKeys[parseInt(e.key, 10) - 1]);
    }
  });

  // Sidebar mini wave animation
  function animateSidebarWave() {
    const c = document.getElementById('sidebarWaveCanvas');
    if (c) {
      const ctx = c.getContext('2d');
      ctx.clearRect(0, 0, c.width, c.height);
      const styles = getComputedStyle(document.documentElement);
      ctx.strokeStyle = styles.getPropertyValue('--cyan-primary').trim() || '#00E5FF';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let x = 0; x < c.width; x += 3) {
        const y = 17 + Math.sin(x * 0.08 + performance.now() * 0.0035) * 6;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    requestAnimationFrame(animateSidebarWave);
  }

  // Window Initialization
  window.addEventListener('DOMContentLoaded', () => {
    let saved = 'light';
    try {
      saved = localStorage.getItem('rj_theme') || 'light';
    } catch (e) {
      saved = 'light';
    }
    window.setAppTheme(saved);

    sim.init();
    renderLiquidityCanvas();
    animateSidebarWave();
    runGsapEntrance('#view-overview');
  });

  window.addEventListener('resize', () => {
    renderLiquidityCanvas();
    renderPortfolioCanvases();
    renderStressCanvas();
  });
})();
