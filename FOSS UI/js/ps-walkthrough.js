/**
 * =========================================================
 * RESERVE*JUGGLER — HIGH-SPEED DIAGRAMMATIC FLOW SIMULATION
 * Minimalist Quantitative Diagram & 60 FPS Particle Physics
 * Zero Fluff: Headings, Live Numbers & Animated Capital Conduits
 * Uniform Project Cyan/Blue Theme (Zero Purple)
 * =========================================================
 */

(function () {
  'use strict';

  const SCENARIOS = [
    {
      step: 0,
      id: 'inflow',
      title: '1. Inbound Inflow',
      headline: '+$20.0M/hr Inbound Crediting to Tier 0 Buffer',
      time: '14:00 UTC',
      flowType: 'inflow_to_tier0',
      balances: { t0: 24810500, t1: 0, t2: 35000000, t3: 15000000 },
      yieldTotal: 0,
      penaltiesSaved: 0,
      summary: 'Continuous $20M/hr inbound clearing. Tier 0 liquid buffer sized to absorb daytime volatility.'
    },
    {
      step: 1,
      id: 'repo_sweep',
      title: '2. 17:00 Sweep',
      headline: '17:00 Central Bank Window: $6.61M Swept to Tier 1 Repo (4.30% APY)',
      time: '17:00 UTC',
      flowType: 'sweep_tier0_to_tier1',
      balances: { t0: 18200000, t1: 6610500, t2: 35000000, t3: 15000000 },
      yieldTotal: 4868.10,
      penaltiesSaved: 0,
      summary: '15-hour overnight lock engages. $18.2M held in buffer for 99.5% night payout safety.'
    },
    {
      step: 2,
      id: 'midnight_shock',
      title: '3. 02:00 Shock',
      headline: '$4.5M Midnight Withdrawal Absorbed by Buffer (Repo 100% Sealed)',
      time: '02:15 UTC',
      flowType: 'shock_tier0_outflow',
      balances: { t0: 13700000, t1: 6610500, t2: 35000000, t3: 15000000 },
      yieldTotal: 4868.10,
      penaltiesSaved: 22500,
      summary: 'Shock hits while repo is locked. Buffer absorbs 100%, dodging 0.5% early-break fee ($22,500 saved).'
    },
    {
      step: 3,
      id: 'repo_settle',
      title: '4. 08:00 Settle',
      headline: 'Repo Matures: $6.61M Principal + $4,868 Interest Returned to Tier 0',
      time: '08:00 UTC',
      flowType: 'return_tier1_to_tier0',
      balances: { t0: 20315368, t1: 0, t2: 35000000, t3: 15000000 },
      yieldTotal: 4868.10,
      penaltiesSaved: 22500,
      summary: 'Overnight lock expires at 08:00. Unlocked capital replenishes Tier 0 for trading day.'
    },
    {
      step: 4,
      id: 'rate_shock',
      title: '5. Rate Shock',
      headline: 'Bond Yield Spikes +45 bps: Duration Hedged (2% Haircut Shielded)',
      time: '11:30 UTC',
      flowType: 'hedge_tier2',
      balances: { t0: 20315368, t1: 0, t2: 35000000, t3: 15000000 },
      yieldTotal: 4868.10,
      penaltiesSaved: 222500,
      summary: 'Duration matched (dP ≈ -Dmod · dy). No forced midday liquidation of bonds ($200k saved).'
    },
    {
      step: 5,
      id: 'svb_run',
      title: '6. SVB 3.2x Run',
      headline: '3.2x Run Cascade: Tier 0 → Tier 1 → Tier 2 Liquidates; Tier 3 Equities Shielded',
      time: '14:45 UTC',
      flowType: 'waterfall_run',
      balances: { t0: 6000000, t1: 0, t2: 26000000, t3: 15000000 },
      yieldTotal: 4868.10,
      penaltiesSaved: 2472500,
      summary: 'Orderly waterfall protects Tier 3 Growth Equities from fire-sale slippage (+$2,250,000 saved).'
    }
  ];

  let currentIdx = 0;
  let autoTimer = null;
  let isAutoRunning = false;
  let animFrameId = null;

  // Real-time canvas physics state
  let canvas = null;
  let ctx = null;
  let width = 0;
  let height = 0;
  let simTime = 0;
  let particles = [];

  // Smoothly interpolated values for 60 FPS rendering
  let currentTier0 = 24810500;
  let currentTier1 = 0;
  let currentTier2 = 35000000;
  let currentTier3 = 15000000;
  let currentYield = 0;
  let currentPenalties = 0;

  // Node layout cache
  let nodes = {};

  function initSimulationCanvas() {
    canvas = document.getElementById('diagSimulationCanvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
  }

  function resizeCanvas() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    width = rect.width;
    height = rect.height;
    if (ctx) ctx.scale(dpr, dpr);
    recalculateNodePositions();
  }

  function recalculateNodePositions() {
    if (!width || !height) return;
    const centerY = height * 0.52;

    nodes = {
      inflow: { x: width * 0.08, y: centerY, label: 'INBOUND', sub: '+$20M/hr' },
      tier0: { x: width * 0.27, y: centerY, label: 'Tier 0: Checking', max: 30000000, apy: '0.0%' },
      tier1: { x: width * 0.51, y: height * 0.25, label: 'Tier 1: Repo', max: 15000000, apy: '4.30%' },
      tier2: { x: width * 0.51, y: height * 0.75, label: 'Tier 2: Sovereign', max: 40000000, apy: '4.80%' },
      tier3: { x: width * 0.74, y: centerY, label: 'Tier 3: Equities', max: 20000000, apy: '10.0%' },
      outflow: { x: width * 0.93, y: centerY, label: 'OUTFLOW', sub: 'Obligations' }
    };
  }

  const CARD_W = 152;
  const CARD_H = 84;

  function getConduitPort(nodeKey, direction) {
    const n = nodes[nodeKey];
    if (!n) return { x: 0, y: 0 };

    if (nodeKey === 'inflow') {
      return { x: n.x + 35, y: n.y };
    }
    if (nodeKey === 'outflow') {
      return { x: n.x - 36, y: n.y };
    }

    if (direction === 'left') {
      return { x: n.x - CARD_W / 2, y: n.y };
    }
    if (direction === 'right') {
      return { x: n.x + CARD_W / 2, y: n.y };
    }
    if (direction === 'top') {
      return { x: n.x, y: n.y - CARD_H / 2 };
    }
    if (direction === 'bottom') {
      return { x: n.x, y: n.y + CARD_H / 2 };
    }
    return { x: n.x, y: n.y };
  }

  function startCanvasLoop() {
    if (animFrameId) cancelAnimationFrame(animFrameId);
    const loop = () => {
      updateSimulationPhysics();
      renderSimulationCanvas();
      animFrameId = requestAnimationFrame(loop);
    };
    animFrameId = requestAnimationFrame(loop);
  }

  function stopCanvasLoop() {
    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  function spawnConduitParticle(fromKey, toKey, color, count = 1, speed = 0.016) {
    if (!nodes[fromKey] || !nodes[toKey]) return;
    for (let i = 0; i < count; i++) {
      particles.push({
        fromKey,
        toKey,
        progress: Math.random() * 0.15,
        speed: speed + Math.random() * 0.008,
        color: color || '#00E5FF',
        size: 3 + Math.random() * 2
      });
    }
  }

  function updateSimulationPhysics() {
    simTime += 0.025;

    const target = SCENARIOS[currentIdx];
    const ease = 0.08;

    // Smooth numerical interpolations
    currentTier0 += (target.balances.t0 - currentTier0) * ease;
    currentTier1 += (target.balances.t1 - currentTier1) * ease;
    currentTier2 += (target.balances.t2 - currentTier2) * ease;
    currentTier3 += (target.balances.t3 - currentTier3) * ease;
    currentYield += (target.yieldTotal - currentYield) * ease;
    currentPenalties += (target.penaltiesSaved - currentPenalties) * ease;

    // Spawn continuous particles based on scenario flow type
    if (Math.random() < 0.45) {
      if (target.flowType === 'inflow_to_tier0') {
        spawnConduitParticle('inflow', 'tier0', '#10B981', 1, 0.02);
      } else if (target.flowType === 'sweep_tier0_to_tier1') {
        spawnConduitParticle('tier0', 'tier1', '#00E5FF', 2, 0.024);
      } else if (target.flowType === 'shock_tier0_outflow') {
        spawnConduitParticle('tier0', 'outflow', '#EF4444', 3, 0.03);
      } else if (target.flowType === 'return_tier1_to_tier0') {
        spawnConduitParticle('tier1', 'tier0', '#10B981', 2, 0.024);
      } else if (target.flowType === 'hedge_tier2') {
        spawnConduitParticle('inflow', 'tier0', '#10B981', 1, 0.015);
        spawnConduitParticle('tier0', 'tier2', '#00E5FF', 1, 0.018);
      } else if (target.flowType === 'waterfall_run') {
        spawnConduitParticle('tier0', 'outflow', '#EF4444', 3, 0.035);
        spawnConduitParticle('tier2', 'outflow', '#F59E0B', 2, 0.025);
      }
    }

    // Update existing particles
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.progress += p.speed;
      if (p.progress >= 1.0) {
        particles.splice(i, 1);
      }
    }

    // Update Top Metric Numbers on HUD
    updateHudNumbers();
  }

  function updateHudNumbers() {
    const elT0 = document.getElementById('diagSimTier0');
    const elT1 = document.getElementById('diagSimTier1');
    const elT2 = document.getElementById('diagSimTier2');
    const elT3 = document.getElementById('diagSimTier3');
    const elYield = document.getElementById('diagSimYield');
    const elPen = document.getElementById('diagSimPenalties');

    if (elT0) elT0.textContent = `$${(currentTier0 / 1e6).toFixed(2)}M`;
    if (elT1) elT1.textContent = `$${(currentTier1 / 1e6).toFixed(2)}M`;
    if (elT2) elT2.textContent = `$${(currentTier2 / 1e6).toFixed(2)}M`;
    if (elT3) elT3.textContent = `$${(currentTier3 / 1e6).toFixed(2)}M`;
    if (elYield) elYield.textContent = `+$${Math.round(currentYield).toLocaleString()}`;
    if (elPen) elPen.textContent = `+$${Math.round(currentPenalties).toLocaleString()}`;
  }

  function renderSimulationCanvas() {
    if (!ctx || !width || !height) return;
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';

    ctx.clearRect(0, 0, width, height);

    const pipeColor = isLight ? 'rgba(0, 0, 0, 0.09)' : 'rgba(255, 255, 255, 0.12)';
    const cyanMain = isLight ? '#0088A3' : '#00E5FF';

    // 1. Draw Connecting Conduit Pipes (Edge-to-Edge Ports so NO lines cross inside card boxes!)
    const conduits = [
      { from: 'inflow', to: 'tier0', fromDir: 'right', toDir: 'left' },
      { from: 'tier0', to: 'tier1', fromDir: 'right', toDir: 'left' },
      { from: 'tier0', to: 'tier2', fromDir: 'right', toDir: 'left' },
      { from: 'tier1', to: 'tier3', fromDir: 'right', toDir: 'left' },
      { from: 'tier2', to: 'tier3', fromDir: 'right', toDir: 'left' },
      { from: 'tier0', to: 'tier3', fromDir: 'right', toDir: 'left' },
      { from: 'tier3', to: 'outflow', fromDir: 'right', toDir: 'left' }
    ];

    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    conduits.forEach(c => {
      const p1 = getConduitPort(c.from, c.fromDir);
      const p2 = getConduitPort(c.to, c.toDir);
      if (!p1 || !p2) return;
      ctx.strokeStyle = pipeColor;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    });

    // 2. Draw Active Moving Particles through Conduit Pipes
    particles.forEach(p => {
      let fromDir = 'right';
      let toDir = 'left';
      if (p.fromKey === 'tier1' && p.toKey === 'tier0') {
        fromDir = 'left';
        toDir = 'right';
      }
      const p1 = getConduitPort(p.fromKey, fromDir);
      const p2 = getConduitPort(p.toKey, toDir);
      if (!p1 || !p2) return;

      const px = p1.x + (p2.x - p1.x) * p.progress;
      const py = p1.y + (p2.y - p1.y) * p.progress;

      ctx.beginPath();
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.arc(px, py, p.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // 3. Render Source & Sink Nodes
    // Inflow Node
    drawPillNode(nodes.inflow.x, nodes.inflow.y, 70, 36, '#10B981', 'INFLOW', '+$20M/hr', isLight);
    // Outflow Node
    const isShock = currentIdx === 2 || currentIdx === 5;
    drawPillNode(nodes.outflow.x, nodes.outflow.y, 72, 36, isShock ? '#EF4444' : '#64748B', isShock ? 'PANIC DRAIN' : 'OUTFLOW', isShock ? '-$64M/hr' : 'Normal', isLight);

    // 4. Render 4 Tier Vault Reservoirs (Clean Formatting, Bounded Text, Zero Overflow)
    drawVaultTank(nodes.tier0, currentTier0, '#10B981', 'Tier 0: Checking', '100% LIQUID', '0.0%', isLight);
    
    // Tier 1 Overnight Repo
    const isRepoLocked = currentIdx === 1 || currentIdx === 2;
    drawVaultTank(nodes.tier1, currentTier1, cyanMain, 'Tier 1: Repo', isRepoLocked ? 'LOCKED [17:00-08:00]' : 'UNLOCKED / SETTLED', '4.30%', isLight, isRepoLocked);

    // Tier 2 Sovereign Debt
    const isHedged = currentIdx === 4;
    drawVaultTank(nodes.tier2, currentTier2, '#F59E0B', 'Tier 2: Sovereign', isHedged ? 'HEDGED (2% Shield)' : 'Duration D=1.85', '4.80%', isLight);

    // Tier 3 Growth Equities (Shielded)
    const isShielded = currentIdx === 5;
    drawVaultTank(nodes.tier3, currentTier3, cyanMain, 'Tier 3: Equities', isShielded ? '100% QUARANTINED' : 'Blue-Chip Index', '10.0%', isLight, false, isShielded);
  }

  function drawPillNode(x, y, w, h, color, text1, text2, isLight) {
    const bg = isLight ? '#FFFFFF' : '#0B101D';
    ctx.save();
    ctx.translate(x, y);

    ctx.shadowColor = isLight ? 'rgba(0, 0, 0, 0.06)' : 'rgba(0, 0, 0, 0.4)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetY = 2;
    ctx.fillStyle = bg;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    roundRect(ctx, -w / 2, -h / 2, w, h, 10, true, true);
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;

    ctx.fillStyle = color;
    ctx.font = 'bold 9.5px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(text1, 0, -2);

    ctx.fillStyle = isLight ? '#475569' : '#94A3B8';
    ctx.font = '8.5px "IBM Plex Mono", monospace';
    ctx.fillText(text2, 0, 11);

    ctx.restore();
  }

  function drawVaultTank(node, balance, color, title, status, apy, isLight, isLocked = false, isQuarantined = false) {
    const W = CARD_W;
    const H = CARD_H;
    const x = node.x - W / 2;
    const y = node.y - H / 2;
    const bg = isLight ? '#FFFFFF' : '#0F172A';
    const textMain = isLight ? '#0A0F1D' : '#FFFFFF';
    const borderCol = isLight ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.18)';

    ctx.save();

    // 1. Protective Glow if locked or quarantined
    if (isLocked || isQuarantined) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      roundRect(ctx, x - 2, y - 2, W + 4, H + 4, 14, false, true);
      ctx.shadowBlur = 0;
    }

    // 2. Card Shadow & Background Container
    ctx.shadowColor = isLight ? 'rgba(0, 0, 0, 0.07)' : 'rgba(0, 0, 0, 0.5)';
    ctx.shadowBlur = 10;
    ctx.shadowOffsetY = 3;
    ctx.fillStyle = bg;
    ctx.strokeStyle = borderCol;
    ctx.lineWidth = 1.2;
    roundRect(ctx, x, y, W, H, 12, true, true);
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;

    // 3. Fluid Level Reservoir at the bottom
    const maxFluidH = 22;
    const fillRatio = Math.min(1.0, Math.max(0.08, balance / node.max));
    const fluidH = maxFluidH * fillRatio;
    const fluidY = y + H - fluidH;

    ctx.save();
    roundRect(ctx, x + 1.5, y + H - maxFluidH - 4, W - 3, maxFluidH + 3, [0, 0, 11, 11], false, false);
    ctx.clip();

    ctx.fillStyle = color;
    ctx.globalAlpha = isLight ? 0.16 : 0.25;
    ctx.fillRect(x + 1.5, fluidY, W - 3, fluidH + 4);

    // Wave surface crest
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.85;
    for (let wx = 0; wx <= W - 3; wx += 4) {
      const wy = fluidY + Math.sin(wx * 0.12 + simTime * 2.5) * 2;
      if (wx === 0) ctx.moveTo(x + 1.5 + wx, wy); else ctx.lineTo(x + 1.5 + wx, wy);
    }
    ctx.stroke();
    ctx.restore();

    // 4. Header: Title (left) & APY (right)
    ctx.fillStyle = color;
    ctx.font = 'bold 10.5px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(title, x + 12, y + 20);

    ctx.font = 'bold 9px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(apy, x + W - 12, y + 20);
    ctx.textAlign = 'left';

    // Thin subtle divider under header
    ctx.strokeStyle = isLight ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + 10, y + 27);
    ctx.lineTo(x + W - 10, y + 27);
    ctx.stroke();

    // 5. Large Prominent Balance Value
    ctx.fillStyle = textMain;
    ctx.font = 'bold 17px "IBM Plex Mono", monospace';
    const balStr = `$${(balance / 1e6).toFixed(2)}M`;
    ctx.fillText(balStr, x + 12, y + 49);

    // 6. Status Footer (above fluid wave)
    ctx.fillStyle = color;
    ctx.font = 'bold 8.5px "IBM Plex Mono", monospace';
    ctx.fillText(status, x + 12, y + 68);

    ctx.restore();
  }

  function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    if (fill) ctx.fill();
    if (stroke) ctx.stroke();
  }

  // Set & Switch Scenario
  window.setDiagScenario = function (idx) {
    if (idx < 0) idx = 0;
    if (idx >= SCENARIOS.length) idx = SCENARIOS.length - 1;
    currentIdx = idx;

    const s = SCENARIOS[currentIdx];

    // Update active tab buttons in UNIFORM BLUE / CYAN
    for (let i = 0; i < SCENARIOS.length; i++) {
      const tab = document.getElementById(`diagTab-${i}`);
      if (tab) {
        if (i === currentIdx) {
          tab.className = 'px-3 py-1.5 rounded-xl text-xs font-mono font-bold bg-[#0088A3] dark:bg-[#00E5FF] text-white dark:text-[#07090E] shadow-sm transform scale-105 transition-all';
        } else if (i < currentIdx) {
          tab.className = 'px-3 py-1.5 rounded-xl text-xs font-mono font-bold text-emerald-500 bg-emerald-500/10 border border-emerald-500/30 transition-all';
        } else {
          tab.className = 'px-3 py-1.5 rounded-xl text-xs font-mono font-medium secondary-text border border-transparent hover:border-slate-300 dark:hover:border-slate-700 transition-all';
        }
      }
    }

    // Headline & Summary updates
    const titleEl = document.getElementById('diagScenarioHeadline');
    const clockEl = document.getElementById('diagScenarioClock');
    const descEl = document.getElementById('diagScenarioDesc');

    if (titleEl) titleEl.textContent = s.headline;
    if (clockEl) clockEl.textContent = s.time;
    if (descEl) descEl.textContent = s.summary;

    const prevBtn = document.getElementById('diagPrevBtn');
    const nextBtn = document.getElementById('diagNextBtn');
    if (prevBtn) prevBtn.disabled = currentIdx === 0;
    if (nextBtn) {
      nextBtn.textContent = currentIdx === SCENARIOS.length - 1 ? 'Restart Flow ↺' : 'Next Scenario →';
    }
  };

  window.diagStepNext = function () {
    if (currentIdx >= SCENARIOS.length - 1) {
      window.setDiagScenario(0);
    } else {
      window.setDiagScenario(currentIdx + 1);
    }
  };

  window.diagStepPrev = function () {
    if (currentIdx > 0) {
      window.setDiagScenario(currentIdx - 1);
    }
  };

  window.toggleDiagAutoPlay = function () {
    if (isAutoRunning) {
      stopAutoPlay();
    } else {
      startAutoPlay();
    }
  };

  function startAutoPlay() {
    isAutoRunning = true;
    const btn = document.getElementById('diagAutoPlayBtn');
    if (btn) {
      btn.innerHTML = '<span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span><span>Auto-Playing (3s)...</span>';
      btn.classList.add('border-cyan-500');
    }
    autoTimer = setInterval(() => {
      window.diagStepNext();
    }, 3200);
  }

  function stopAutoPlay() {
    isAutoRunning = false;
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = null;
    const btn = document.getElementById('diagAutoPlayBtn');
    if (btn) {
      btn.innerHTML = '<span>Auto-Play Continuous (3s)</span>';
      btn.classList.remove('border-cyan-500');
    }
  }

  // Open & Close Modal
  window.openPsWalkthroughModal = function (stepIdx = 0) {
    window.ensurePsWalkthroughModal();
    const modal = document.getElementById('psWalkthroughModal');
    if (!modal) return;
    modal.classList.remove('hidden');
    modal.classList.add('flex');

    initSimulationCanvas();
    startCanvasLoop();
    window.setDiagScenario(stepIdx);
  };

  window.closePsWalkthroughModal = function () {
    const modal = document.getElementById('psWalkthroughModal');
    if (!modal) return;
    modal.classList.add('hidden');
    modal.classList.remove('flex');
    stopAutoPlay();
    stopCanvasLoop();
  };

  // Keyboard navigation
  window.addEventListener('keydown', (e) => {
    const modal = document.getElementById('psWalkthroughModal');
    if (!modal || modal.classList.contains('hidden')) return;
    if (e.key === 'Escape') {
      window.closePsWalkthroughModal();
    } else if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      window.diagStepNext();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      window.diagStepPrev();
    }
  });

  // Inject Minimalist Modal Markup (Only Diagrams, Numbers & Headings)
  window.ensurePsWalkthroughModal = function () {
    let modal = document.getElementById('psWalkthroughModal');
    if (modal) {
      modal.remove();
    }

    modal = document.createElement('div');
    modal.id = 'psWalkthroughModal';
    modal.className = 'fixed inset-0 z-50 hidden items-center justify-center p-2 sm:p-5 modal-overlay';
    modal.innerHTML = `
      <div class="curved-card max-w-5xl w-full max-h-[96vh] flex flex-col p-4 sm:p-6 space-y-3.5 overflow-y-auto">
        
        <!-- Header: Minimal Title & Controls -->
        <div class="flex items-center justify-between border-b pb-3" style="border-color: var(--border-color);">
          <div class="flex items-center gap-2.5">
            <span class="pill-badge pill-badge-cyan font-bold text-[10px]">
              Autonomous Flow Simulation
            </span>
            <div class="flex items-center gap-2">
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span class="text-xs font-mono font-bold primary-text">60 FPS Conformal Engine</span>
            </div>
          </div>
          <button onclick="closePsWalkthroughModal()" class="curved-btn py-1 px-3 text-xs font-bold" title="Close [ESC]">&times; Close</button>
        </div>

        <!-- 6 Step Timeline Stepper Tabs (Uniform Project Blue) -->
        <div class="grid grid-cols-3 sm:grid-cols-6 gap-2">
          ${SCENARIOS.map((s, i) => `
            <button onclick="setDiagScenario(${i})" id="diagTab-${i}" class="px-2 py-1.5 rounded-xl text-xs font-mono font-bold text-center truncate transition-all">
              ${s.title}
            </button>
          `).join('')}
        </div>

        <!-- Active Scenario Headline Banner -->
        <div class="flex items-center justify-between px-3.5 py-2 rounded-xl border text-xs font-mono" style="border-color: var(--border-color); background: var(--bg-card);">
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full pulse-dot" style="background-color: var(--cyan-primary);"></span>
            <span id="diagScenarioHeadline" class="font-bold heading-text text-xs sm:text-sm">
              1. Inbound Inflow
            </span>
          </div>
          <div class="flex items-center gap-2">
            <span class="muted-text">Sim Clock:</span>
            <span id="diagScenarioClock" class="font-bold text-cyan-500">14:00 UTC</span>
          </div>
        </div>

        <!-- HUD Strip: Live Rolling Numbers & Metrics Only -->
        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          <div class="p-2.5 rounded-xl border text-xs font-mono" style="border-color: var(--border-color); background: var(--bg-card);">
            <span class="text-[9px] muted-text uppercase block leading-none">Tier 0 Checking</span>
            <span id="diagSimTier0" class="text-sm font-bold text-emerald-500 mt-1 block">$24.81M</span>
          </div>
          <div class="p-2.5 rounded-xl border text-xs font-mono" style="border-color: var(--border-color); background: var(--bg-card);">
            <span class="text-[9px] muted-text uppercase block leading-none">Tier 1 Repo</span>
            <span id="diagSimTier1" class="text-sm font-bold text-cyan-500 mt-1 block">$0.00M</span>
          </div>
          <div class="p-2.5 rounded-xl border text-xs font-mono" style="border-color: var(--border-color); background: var(--bg-card);">
            <span class="text-[9px] muted-text uppercase block leading-none">Tier 2 Sovereign</span>
            <span id="diagSimTier2" class="text-sm font-bold text-amber-500 mt-1 block">$35.00M</span>
          </div>
          <div class="p-2.5 rounded-xl border text-xs font-mono" style="border-color: var(--border-color); background: var(--bg-card);">
            <span class="text-[9px] muted-text uppercase block leading-none">Tier 3 Equities</span>
            <span id="diagSimTier3" class="text-sm font-bold text-cyan-600 dark:text-cyan-400 mt-1 block">$15.00M</span>
          </div>
          <div class="p-2.5 rounded-xl border text-xs font-mono" style="border-color: var(--border-color); background: var(--bg-card);">
            <span class="text-[9px] muted-text uppercase block leading-none">Overnight Yield</span>
            <span id="diagSimYield" class="text-sm font-bold text-cyan-500 mt-1 block">+$0</span>
          </div>
          <div class="p-2.5 rounded-xl border text-xs font-mono" style="border-color: var(--border-color); background: var(--bg-card);">
            <span class="text-[9px] muted-text uppercase block leading-none">Penalties Dodged</span>
            <span id="diagSimPenalties" class="text-sm font-bold text-emerald-500 mt-1 block">+$0</span>
          </div>
        </div>

        <!-- ================= THE CORE SIMULATION CANVAS ================= -->
        <div class="w-full h-80 rounded-2xl border relative overflow-hidden" style="border-color: var(--border-color); background-color: var(--bg-card);">
          <canvas id="diagSimulationCanvas" class="w-full h-full block"></canvas>
        </div>

        <!-- Compact 1-Line Status -->
        <div class="p-2.5 rounded-xl border text-xs font-mono flex items-center gap-2" style="border-color: var(--border-color); background: var(--bg-card);">
          <span class="font-bold text-cyan-500 flex-shrink-0">[DECISION]:</span>
          <span id="diagScenarioDesc" class="secondary-text truncate">
            Continuous $20M/hr inbound clearing. Tier 0 liquid buffer sized to absorb daytime volatility.
          </span>
        </div>

        <!-- Footer Action Buttons (Uniform Project Blue) -->
        <div class="flex items-center justify-between gap-3 pt-2 border-t" style="border-color: var(--border-color);">
          <button onclick="toggleDiagAutoPlay()" id="diagAutoPlayBtn" class="curved-btn py-1.5 px-3 text-xs font-mono font-bold flex items-center gap-1.5">
            <span>Auto-Play Continuous (3s)</span>
          </button>

          <div class="flex items-center gap-2">
            <button onclick="diagStepPrev()" id="diagPrevBtn" class="curved-btn py-1.5 px-4 text-xs font-mono font-bold" disabled>
              &larr; Prev Step
            </button>
            <button onclick="diagStepNext()" id="diagNextBtn" class="demo-flow-btn py-1.5 px-5 text-xs font-mono font-bold">
              Next Scenario &rarr;
            </button>
          </div>
        </div>

      </div>
    `;
    document.body.appendChild(modal);
  };

  window.addEventListener('DOMContentLoaded', () => {
    window.ensurePsWalkthroughModal();
  });
})();
