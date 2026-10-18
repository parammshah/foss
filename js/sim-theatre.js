/**
 * =========================================================
 * RESERVE*JUGGLER — HIGH-FIDELITY SIMULATION THEATRE ENGINE
 * Real-time 60 FPS Multi-Tier Canvas Fluid Simulation
 * Scenario Triggers, Particle Flow Conduits & Telemetry HUD
 * =========================================================
 */

(function () {
  'use strict';

  class SimTheatreController {
    constructor() {
      this.canvas = null;
      this.ctx = null;
      this.isRunning = false;
      this.animFrameId = null;
      this.speed = 1.0;
      this.simTime = 0;
      this.simHour = 17; // Starts at 17:00 UTC
      this.simMinute = 0;
      
      // Tier Balances & States
      this.tier0 = 24810500;   // Checking Buffer
      this.tier1 = 0;          // Overnight Repo
      this.tier2 = 35000000;   // 2Y Sovereign Bonds
      this.tier3 = 15000000;   // Growth Equities
      
      // Target balances for smooth interpolation
      this.targetTier0 = 24810500;
      this.targetTier1 = 0;
      this.targetTier2 = 35000000;
      this.targetTier3 = 15000000;

      this.repoLocked = false;
      this.yieldEarned = 0;
      this.penaltiesAvoided = 0;
      this.activeScenario = 'Normal Flow (SEC 2a-7)';
      this.particles = [];
      this.maxParticles = 65;

      this.decisionLogs = [
        { time: '16:00:00', text: 'Telemetry initialized. Baseline $20.0M/hr cash flow clearing.' }
      ];
    }

    init() {
      this.canvas = document.getElementById('simTheatreCanvas');
      if (!this.canvas) return;
      this.ctx = this.canvas.getContext('2d');
      this.resize();
      window.addEventListener('resize', () => this.resize());
    }

    resize() {
      if (!this.canvas) return;
      const rect = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.width = rect.width;
      this.height = rect.height;
      this.ctx.scale(dpr, dpr);
    }

    start() {
      if (this.isRunning) return;
      this.isRunning = true;
      this.resize();
      const loop = () => {
        if (!this.isRunning) return;
        this.update();
        this.render();
        this.animFrameId = requestAnimationFrame(loop);
      };
      this.animFrameId = requestAnimationFrame(loop);
      this.updateHud();
    }

    stop() {
      this.isRunning = false;
      if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    }

    setSpeed(s) {
      this.speed = s;
      const btns = ['simSpeed1', 'simSpeed5', 'simSpeed20'];
      btns.forEach(b => {
        const el = document.getElementById(b);
        if (el) el.classList.remove('bg-cyan-500/20', 'text-cyan-400', 'font-bold');
      });
      const target = document.getElementById(`simSpeed${s}`);
      if (target) target.classList.add('bg-cyan-500/20', 'text-cyan-400', 'font-bold');
    }

    triggerScenario(scenarioKey) {
      if (scenarioKey === 'repo_sweep') {
        this.activeScenario = '17:00 UTC Repo Window Sweep';
        this.simHour = 17;
        this.simMinute = 0;
        this.repoLocked = true;
        // Lock $18.20M in Tier 0, sweep surplus into Tier 1
        const surplus = Math.max(0, this.targetTier0 - 18200000);
        this.targetTier0 = 18200000;
        this.targetTier1 = surplus;
        this.spawnBurst(0, 1, 24, '#00E5FF');
        this.logDecision('17:00:00', `Locked $18.20M conformal cushion in Tier 0. Swept $${(surplus/1e6).toFixed(2)}M surplus into Tier 1 Repo @ 4.30% APY.`);
      } else if (scenarioKey === 'midnight_shock') {
        this.activeScenario = '02:00 UTC Midnight Withdrawal Spike';
        this.simHour = 2;
        this.simMinute = 15;
        // $4.5M withdrawal absorbed 100% in Tier 0 buffer
        this.targetTier0 = Math.max(10000000, this.targetTier0 - 4500000);
        this.penaltiesAvoided += 22500; // 0.5% of $4.5M
        this.spawnBurst(0, 'exit', 28, '#F43F5E');
        this.logDecision('02:15:00', 'Absorbed $4.5M withdrawal wave in Tier 0 buffer. Repo vault kept sealed \u2192 $22,500 break fee dodged!');
      } else if (scenarioKey === 'repo_settle') {
        this.activeScenario = '08:00 UTC Repo Release & Settlement';
        this.simHour = 8;
        this.simMinute = 0;
        this.repoLocked = false;
        const interest = 4868.10;
        this.yieldEarned += interest;
        this.targetTier0 += (this.targetTier1 + interest);
        this.targetTier1 = 0;
        this.spawnBurst(1, 0, 24, '#10B981');
        this.logDecision('08:00:00', `08:00 Settlement completed. Released principal + $${interest.toFixed(2)} overnight interest back to Tier 0.`);
      } else if (scenarioKey === 'rate_shock') {
        this.activeScenario = 'Sovereign Debt Rate Shift (+45 bps)';
        this.simHour = 11;
        this.penaltiesAvoided += 200000; // Avoided 2% bond haircut
        this.spawnBurst(2, 2, 20, '#F59E0B');
        this.logDecision('11:00:00', 'Yield shift (+45 bps) detected. Duration-matching protocol rebalanced Tier 2 \u2192 2.0% haircut ($200K) prevented.');
      } else if (scenarioKey === 'svb_run') {
        this.activeScenario = 'SVB 3.2x Run & Waterfall Defense';
        this.simHour = 14;
        this.simMinute = 40;
        // Cascading waterfall: Tier 0 -> Tier 1 -> Tier 2
        this.targetTier0 = 6000000;
        this.targetTier1 = 0;
        this.targetTier2 = 26000000;
        this.penaltiesAvoided += 2250000; // 15% slippage on equities saved
        this.spawnBurst(0, 'exit', 35, '#F43F5E');
        this.spawnBurst(2, 0, 15, '#F59E0B');
        this.logDecision('14:40:00', 'Cascading waterfall engaged: Tier 0 \u2192 Tier 1 \u2192 Tier 2 tapped. Tier 3 Equities quarantined \u2192 $2.25M fire-sale slippage saved!');
      }
      this.updateHud();
    }

    logDecision(time, text) {
      this.decisionLogs.unshift({ time, text });
      if (this.decisionLogs.length > 6) this.decisionLogs.pop();
      const listEl = document.getElementById('simDecisionList');
      if (listEl) {
        listEl.innerHTML = this.decisionLogs.map(l => `
          <div class="flex items-start gap-2 py-1.5 border-b text-[11px] font-mono" style="border-color: var(--border-color);">
            <span class="font-bold text-cyan-500 flex-shrink-0">[${l.time}]</span>
            <span class="secondary-text leading-tight">${l.text}</span>
          </div>
        `).join('');
      }
    }

    spawnBurst(fromTier, toTier, count, color) {
      for (let i = 0; i < count; i++) {
        this.particles.push({
          from: fromTier,
          to: toTier,
          progress: Math.random() * 0.2,
          speed: 0.012 + Math.random() * 0.015,
          color: color || '#00E5FF',
          size: 2.2 + Math.random() * 2.5
        });
      }
    }

    update() {
      this.simTime += 0.02 * this.speed;

      // Advance clock smoothly
      this.simMinute += 0.4 * this.speed;
      if (this.simMinute >= 60) {
        this.simMinute = 0;
        this.simHour = (this.simHour + 1) % 24;
      }

      // Smooth interpolation of balances
      this.tier0 += (this.targetTier0 - this.tier0) * 0.06;
      this.tier1 += (this.targetTier1 - this.tier1) * 0.06;
      this.tier2 += (this.targetTier2 - this.tier2) * 0.06;
      this.tier3 += (this.targetTier3 - this.tier3) * 0.06;

      // Update particles
      for (let i = this.particles.length - 1; i >= 0; i--) {
        const p = this.particles[i];
        p.progress += p.speed * this.speed;
        if (p.progress >= 1.0) {
          this.particles.splice(i, 1);
        }
      }

      // Routine flow particles
      if (Math.random() < 0.25) {
        this.particles.push({
          from: 'inflow',
          to: 0,
          progress: 0,
          speed: 0.015 + Math.random() * 0.01,
          color: '#10B981',
          size: 2.5
        });
      }

      this.updateHud();
    }

    updateHud() {
      const clockEl = document.getElementById('simClockHud');
      const scnEl = document.getElementById('simScenarioName');
      const b0 = document.getElementById('simHudTier0');
      const b1 = document.getElementById('simHudTier1');
      const b2 = document.getElementById('simHudTier2');
      const b3 = document.getElementById('simHudTier3');
      const yld = document.getElementById('simHudYield');
      const pen = document.getElementById('simHudPenalties');

      const hh = String(this.simHour).padStart(2, '0');
      const mm = String(Math.floor(this.simMinute)).padStart(2, '0');
      if (clockEl) clockEl.textContent = `${hh}:${mm} UTC`;
      if (scnEl) scnEl.textContent = this.activeScenario;

      if (b0) b0.textContent = `$${(this.tier0 / 1e6).toFixed(2)}M`;
      if (b1) b1.textContent = `$${(this.tier1 / 1e6).toFixed(2)}M`;
      if (b2) b2.textContent = `$${(this.tier2 / 1e6).toFixed(2)}M`;
      if (b3) b3.textContent = `$${(this.tier3 / 1e6).toFixed(2)}M`;

      if (yld) yld.textContent = `+$${Math.round(this.yieldEarned).toLocaleString()}`;
      if (pen) pen.textContent = `+$${Math.round(this.penaltiesAvoided).toLocaleString()}`;
    }

    render() {
      if (!this.ctx || !this.width) return;
      const ctx = this.ctx;
      const W = this.width;
      const H = this.height;

      ctx.clearRect(0, 0, W, H);

      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      const borderCol = isLight ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.1)';
      const cardBg = isLight ? '#FFFFFF' : '#0B101D';
      const textMain = isLight ? '#0A0F1D' : '#FFFFFF';
      const textMuted = isLight ? '#64748B' : '#94A3B8';

      // 4 Tier Vault Boxes Layout
      const vaultWidth = (W - 80) / 4;
      const vaultHeight = H * 0.62;
      const vaultY = H * 0.28;

      const vaults = [
        {
          idx: 0,
          name: 'Tier 0: Cash Buffer',
          sub: 'Operational Checking',
          apy: '0.0% APY',
          color: '#10B981',
          bal: this.tier0,
          max: 35000000,
          status: '100% Liquid'
        },
        {
          idx: 1,
          name: 'Tier 1: Overnight Repo',
          sub: 'Central Bank Facility',
          apy: '4.3% APY',
          color: '#00E5FF',
          bal: this.tier1,
          max: 20000000,
          status: this.repoLocked ? 'LOCKED (17:00-08:00)' : 'UNLOCKED / OPEN'
        },
        {
          idx: 2,
          name: 'Tier 2: 2Y Sovereign Bonds',
          sub: 'Short-Term Gov Debt',
          apy: '4.8% APY',
          color: '#F59E0B',
          bal: this.tier2,
          max: 45000000,
          status: 'Duration dP \u2248 -D\u00b7dy'
        },
        {
          idx: 3,
          name: 'Tier 3: Growth Equities',
          sub: 'Blue-Chip Index',
          apy: '10.0% APY',
          color: '#8B5CF6',
          bal: this.tier3,
          max: 25000000,
          status: '15% Slippage Shield'
        }
      ];

      // Draw Conduit Pipes between Vaults
      ctx.strokeStyle = borderCol;
      ctx.lineWidth = 3;
      ctx.beginPath();
      // Inflow to Tier 0
      ctx.moveTo(20, vaultY + vaultHeight * 0.5);
      ctx.lineTo(W - 20, vaultY + vaultHeight * 0.5);
      ctx.stroke();

      // Render Vault Cards
      vaults.forEach((v, i) => {
        const x = 20 + i * (vaultWidth + 13);
        const y = vaultY;
        v.x = x + vaultWidth * 0.5;
        v.y = y + vaultHeight * 0.5;

        // Card Container
        ctx.fillStyle = cardBg;
        ctx.strokeStyle = borderCol;
        ctx.lineWidth = 1;
        this.roundRect(ctx, x, y, vaultWidth, vaultHeight, 14, true, true);

        // Fill Level Ratio
        const fillPct = Math.min(1.0, Math.max(0.08, v.bal / v.max));
        const liquidH = (vaultHeight - 75) * fillPct;
        const liquidY = y + vaultHeight - liquidH - 8;

        // Dynamic Fluid Reservoir with wave oscillation
        ctx.save();
        this.roundRect(ctx, x + 6, y + 55, vaultWidth - 12, vaultHeight - 65, 8, false, false);
        ctx.clip();

        ctx.fillStyle = v.color;
        ctx.globalAlpha = isLight ? 0.12 : 0.22;
        ctx.fillRect(x + 6, liquidY, vaultWidth - 12, liquidH + 10);

        // Wave crest
        ctx.beginPath();
        ctx.strokeStyle = v.color;
        ctx.lineWidth = 1.8;
        ctx.globalAlpha = 0.85;
        for (let wx = 0; wx <= vaultWidth - 12; wx += 4) {
          const wy = liquidY + Math.sin(wx * 0.08 + this.simTime * 2 + i) * 3;
          if (wx === 0) ctx.moveTo(x + 6 + wx, wy); else ctx.lineTo(x + 6 + wx, wy);
        }
        ctx.stroke();
        ctx.restore();

        // Vault Header
        ctx.fillStyle = textMain;
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.fillText(v.name, x + 12, y + 22);

        ctx.fillStyle = textMuted;
        ctx.font = '10px "IBM Plex Mono", monospace';
        ctx.fillText(v.sub, x + 12, y + 36);

        // APY Badge
        ctx.fillStyle = v.color;
        ctx.font = 'bold 10px "IBM Plex Mono", monospace';
        ctx.fillText(v.apy, x + vaultWidth - 62, y + 22);

        // Balance Big Metric
        ctx.fillStyle = textMain;
        ctx.font = 'bold 15px "IBM Plex Mono", monospace';
        const balStr = `$${(v.bal / 1e6).toFixed(2)}M`;
        ctx.fillText(balStr, x + 12, y + vaultHeight - 28);

        // Status Tag
        ctx.fillStyle = v.color;
        ctx.font = '9px "IBM Plex Mono", monospace';
        ctx.fillText(v.status, x + 12, y + vaultHeight - 12);
      });

      // Render Animated Liquidity Particles
      this.particles.forEach(p => {
        let sx = 20, sy = vaultY + vaultHeight * 0.5;
        let dx = vaults[0].x, dy = vaults[0].y;

        if (p.from === 'inflow') {
          sx = 10; sy = vaultY + vaultHeight * 0.5;
          dx = vaults[0].x; dy = vaults[0].y;
        } else if (typeof p.from === 'number' && typeof p.to === 'number') {
          sx = vaults[p.from].x; sy = vaults[p.from].y;
          dx = vaults[p.to].x; dy = vaults[p.to].y;
        } else if (p.to === 'exit') {
          sx = vaults[0].x; sy = vaults[0].y;
          dx = W - 10; dy = vaultY + vaultHeight * 0.5;
        }

        const cx = sx + (dx - sx) * p.progress;
        const cy = sy + (dy - sy) * p.progress + Math.sin(p.progress * Math.PI) * -16;

        ctx.beginPath();
        ctx.fillStyle = p.color;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 6;
        ctx.arc(cx, cy, p.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      });
    }

    roundRect(ctx, x, y, width, height, radius, fill, stroke) {
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
  }

  window.SimTheatreController = SimTheatreController;
  window.simTheatreInstance = new SimTheatreController();

  // Unified Simulation Modal
  window.openSimulationModal = function (scenario = 'repo_sweep') {
    if (window.openPsWalkthroughModal) {
      const scenarioMap = {
        'repo_sweep': 1,
        'midnight_shock': 2,
        'repo_settle': 3,
        'rate_shock': 4,
        'svb_run': 5,
        'night_sweep': 1
      };
      const idx = scenarioMap[scenario] !== undefined ? scenarioMap[scenario] : 1;
      window.openPsWalkthroughModal(idx);
    }
  };

  window.closeSimulationModal = function () {
    if (window.closePsWalkthroughModal) {
      window.closePsWalkthroughModal();
    }
  };

  window.openLandingSimTheatreModal = function () {
    window.openSimulationModal('repo_sweep');
  };
})();
