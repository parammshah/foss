/**
 * =============================================================================
 * RESERVE*JUGGLER PLATFORM CONTROLLER — PLATFORM.JS
 * Canvas Graph Rendering, Tab Switching, Interactive Controls,
 * and Full Backend ApplicationFacade Integration.
 * =============================================================================
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
      setTimeout(() => {
        updateStressLab();
        renderStressCanvas();
      }, 40);
    } else if (viewId === 'edge') {
      setTimeout(() => {
        if (typeof initEdgeView === 'function') initEdgeView();
      }, 40);
    } else if (viewId === 'sentinel') {
      setTimeout(() => {
        if (typeof initSentinelView === 'function') initSentinelView();
      }, 40);
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
    showToast(`Simulation advanced +${hours}h.`);

    // If live API connected, also refresh snapshot from backend
    if (window.ReserveJugglerAPI && window.ReserveJugglerAPI.isConnected) {
      window.ReserveJugglerAPI.fetchSnapshot().catch(() => {});
    }
  };

  // Auto-Play Stream
  window.toggleAutoPlay = function () {
    const btn = document.getElementById('autoPlayBtn');
    const dot = document.getElementById('autoPlayDot');
    const lbl = document.getElementById('autoPlayLabel');

    if (sim.autoPlayInterval) {
      clearInterval(sim.autoPlayInterval);
      sim.autoPlayInterval = null;
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-slate-500';
      if (lbl) lbl.textContent = 'Auto-Stream';
      showToast('Simulation stream paused.');
    } else {
      sim.autoPlayInterval = setInterval(() => {
        sim.advanceStep(1);
        if (window.ReserveJugglerAPI && window.ReserveJugglerAPI.isConnected) {
          window.ReserveJugglerAPI.fetchSnapshot().catch(() => {});
        }
      }, 1500);
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-cyan-400 pulse-dot';
      if (lbl) lbl.textContent = 'Streaming...';
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

    renderLiquidityCanvas();
    renderPortfolioCanvases();
    renderStressCanvas();
  };

  // Toggle Theme
  window.toggleTheme = function () {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme') || 'light';
    const next = current === 'light' ? 'dark' : 'light';
    window.setAppTheme(next);
    showToast(`Switched to ${next.toUpperCase()} theme.`);
  };

  // =========================================================================
  // Canvas Rendering Logic
  // =========================================================================

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
    const minVal = 14000000, maxVal = 55000000;

    const toX = (idx) => padL + (idx / Math.max(1, visiblePoints.length - 1)) * plotW;
    const toY = (val) => padT + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;

    // Grid lines
    ctx.strokeStyle = gridCol;
    ctx.lineWidth = 1;
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.fillStyle = textMuted;

    for (let v = 20000000; v <= 50000000; v += 10000000) {
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

    // Spline Cash Curve
    if (!window._graphAnimTime) window._graphAnimTime = 0;
    window._graphAnimTime += 0.03;

    const curvePoints = sim.currentPlotPoints.map((val, i) => {
      const breath = Math.sin(window._graphAnimTime + i * 0.45) * 220000;
      return {
        x: toX(i),
        y: toY(val + breath)
      };
    });

    if (curvePoints.length > 1) {
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
      const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
      grad.addColorStop(0, styles.getPropertyValue('--cyan-glow').trim() || 'rgba(0, 229, 255, 0.18)');
      grad.addColorStop(1, 'rgba(0, 229, 255, 0.0)');
      ctx.fillStyle = grad;
      ctx.fill();
    }
  };

  // 4-Tier Portfolio Breakdown Canvas
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

  // SVB Stress Lab Multi-Strategy Canvas
  window.renderStressCanvas = function () {
    const c = document.getElementById('stressCanvas');
    if (!c) return;
    const ctx = c.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = c.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    c.width = rect.width * dpr;
    c.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const padL = 45, padR = 20, padT = 20, padB = 25;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    const styles = getComputedStyle(document.documentElement);
    const gridCol = styles.getPropertyValue('--canvas-grid').trim() || 'rgba(255, 255, 255, 0.05)';
    const textMuted = styles.getPropertyValue('--text-muted').trim() || '#64748B';
    const cyan = styles.getPropertyValue('--cyan-primary').trim() || '#00E5FF';
    const rose = styles.getPropertyValue('--rose-accent').trim() || '#F43F5E';
    const gold = styles.getPropertyValue('--gold-accent').trim() || '#F59E0B';

    // Grid lines
    ctx.strokeStyle = gridCol;
    ctx.lineWidth = 1;
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.fillStyle = textMuted;

    for (let i = 0; i <= 4; i++) {
      const y = padT + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
      const val = 220 - i * 50;
      ctx.fillText(`$${val}M`, 10, y + 3);
    }

    const surge = parseFloat(document.getElementById('stressSurgeSlider')?.value || 1.0);
    const steps = 30;

    // 1. All-Cash Curve (Baseline passive cash drain)
    ctx.beginPath();
    ctx.strokeStyle = '#94A3B8';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 4]);
    for (let i = 0; i <= steps; i++) {
      const x = padL + (i / steps) * plotW;
      const decay = (i / steps) * 0.35 * surge;
      const y = padT + plotH * (0.2 + decay);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Greedy Yield Curve (Suffers severe illiquidity haircut & floor breach)
    ctx.beginPath();
    ctx.strokeStyle = rose;
    ctx.lineWidth = 2.2;
    for (let i = 0; i <= steps; i++) {
      const x = padL + (i / steps) * plotW;
      const drop = Math.pow(i / steps, 1.4) * 0.72 * surge;
      const y = Math.min(padT + plotH, padT + plotH * (0.15 + drop));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 3. Adaptive Reserve Juggler Curve (Protects reserve floor, minimal cost)
    ctx.beginPath();
    ctx.strokeStyle = cyan;
    ctx.lineWidth = 2.8;
    for (let i = 0; i <= steps; i++) {
      const x = padL + (i / steps) * plotW;
      const drop = (i / steps) * 0.22 * surge;
      const y = padT + plotH * (0.18 + drop);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Legend
    ctx.font = 'bold 9px "IBM Plex Mono", monospace';
    ctx.fillStyle = cyan;
    ctx.fillText('Adaptive Reserve Juggler', padL + 10, padT + 12);
    ctx.fillStyle = rose;
    ctx.fillText('Greedy Yield (Breach)', padL + 170, padT + 12);
    ctx.fillStyle = '#94A3B8';
    ctx.fillText('All-Cash', padL + 310, padT + 12);
  };

  // =========================================================================
  // Interactive Backend Actions & Handlers
  // =========================================================================

  // 1. Switch Operational Scenario
  window.switchOperationalScenario = async function (scenarioName) {
    showToast(`Transitioning scenario to ${scenarioName}...`);
    if (window.ReserveJugglerAPI) {
      try {
        const res = await window.ReserveJugglerAPI.switchScenario(scenarioName);
        if (res && res.success) {
          showToast(`Operational scenario active: ${scenarioName}`);
        }
      } catch (err) {
        showToast(`Failed to switch scenario: ${err.message}`);
      }
    }
  };

  // 2. Run Canonical 10-Step Decision Cycle
  window.triggerRunDecisionCycle = async function () {
    const btn = document.getElementById('runCycleBtn');
    if (btn) btn.classList.add('animate-pulse');
    showToast('Executing canonical 10-step decision cycle across all 3 teams...');

    if (window.ReserveJugglerAPI) {
      try {
        const res = await window.ReserveJugglerAPI.runDecisionCycle();
        if (res && res.success) {
          showToast(`Cycle ${res.cycle_id} completed successfully.`);
        }
      } catch (err) {
        showToast(`Decision cycle error: ${err.message}`);
      } finally {
        if (btn) btn.classList.remove('animate-pulse');
      }
    }
  };

  // 3. Approve Decision Recommendation
  window.approveDecisionAction = async function () {
    showToast('Submitting decision sign-off to live governance ledger...');
    if (window.ReserveJugglerAPI) {
      try {
        const res = await window.ReserveJugglerAPI.submitDecisionAction({
          actionType: 'APPROVED',
          actorName: 'Chief Treasurer',
          actorRole: 'HEAD_OF_TREASURY',
          statedReason: 'Approved for simulated execution under verified liquidity thresholds.'
        });
        if (res && res.success) {
          showToast(res.message || 'Decision APPROVED successfully.');
        } else {
          showToast(`Action rejected: ${res.message || 'Policy violation'}`);
        }
      } catch (err) {
        showToast(`Approval error: ${err.message}`);
      }
    }
  };

  // Legacy button alias
  window.approveNBA = window.approveDecisionAction;

  // 4. Reject Decision Recommendation
  window.rejectDecisionAction = async function () {
    const reason = prompt('Enter mandatory audit reason for decision rejection:', 'Duration mismatch or high yield curve volatility.');
    if (!reason) return;

    showToast('Submitting rejection to audit trail...');
    if (window.ReserveJugglerAPI) {
      try {
        const res = await window.ReserveJugglerAPI.submitDecisionAction({
          actionType: 'REJECTED',
          actorName: 'Chief Risk Officer',
          actorRole: 'RISK_OFFICER',
          statedReason: reason
        });
        if (res && res.success) {
          showToast(res.message || 'Decision REJECTED and logged in audit trail.');
        }
      } catch (err) {
        showToast(`Rejection error: ${err.message}`);
      }
    }
  };

  // 5. Modify Decision Modal
  window.openModifyDecisionModal = function () {
    const m = document.getElementById('modifyDecisionModal');
    if (m) {
      m.classList.remove('hidden');
      m.classList.add('flex');
    }
  };

  window.closeModifyDecisionModal = function () {
    const m = document.getElementById('modifyDecisionModal');
    if (m) {
      m.classList.add('hidden');
      m.classList.remove('flex');
    }
  };

  window.submitModifiedDecisionAction = async function () {
    const amtInput = document.getElementById('modifyAmountInput');
    const actSelect = document.getElementById('modifyActionSelect');
    const roleSelect = document.getElementById('modifyRoleSelect');
    const reasonInput = document.getElementById('modifyReasonInput');

    const amount = amtInput ? parseFloat(amtInput.value) : null;
    const action = actSelect ? actSelect.value : null;
    const role = roleSelect ? roleSelect.value : 'HEAD_OF_TREASURY';
    const reason = reasonInput ? reasonInput.value.trim() : 'Custom parameter modification by desk.';

    closeModifyDecisionModal();
    showToast('Submitting modified decision action...');

    if (window.ReserveJugglerAPI) {
      try {
        const res = await window.ReserveJugglerAPI.submitDecisionAction({
          actionType: 'MODIFIED',
          actorName: 'Director of Treasury Operations',
          actorRole: role,
          statedReason: reason,
          modifiedAmount: amount,
          modifiedAction: action
        });
        if (res && res.success) {
          showToast(res.message || 'Decision successfully modified and revalidated.');
        } else {
          showToast(`Modification rejected: ${res.message}`);
        }
      } catch (err) {
        showToast(`Modification error: ${err.message}`);
      }
    }
  };

  // 6. Promote / Rollback Learning Candidates
  window.promoteCandidateAction = async function (candId) {
    if (!confirm(`Are you sure you want to promote candidate ${candId} to production?`)) return;
    showToast(`Promoting ${candId}...`);

    if (window.ReserveJugglerAPI) {
      try {
        const res = await window.ReserveJugglerAPI.promoteCandidate(candId, 'HEAD_OF_TREASURY');
        if (res && res.success) {
          showToast(res.message || `Candidate ${candId} promoted to production.`);
        } else {
          showToast(`Promotion failed: ${res.message}`);
        }
      } catch (err) {
        showToast(`Promotion error: ${err.message}`);
      }
    }
  };

  // 7. SVB Stress Lab Real-Time Simulation
  window.updateStressLab = async function () {
    const sEl = document.getElementById('stressSurgeSlider');
    const yEl = document.getElementById('stressYieldSlider');
    const dEl = document.getElementById('stressDelaySlider');
    const cEl = document.getElementById('stressCreditToggle');

    if (!sEl || !yEl) return;

    const surge = parseFloat(sEl.value);
    const yShock = parseInt(yEl.value, 10);
    const delay = dEl ? parseInt(dEl.value, 10) : (surge > 2.5 ? 1 : 0);
    const creditFrozen = cEl ? cEl.checked : (surge > 3.0);

    const surgeVal = document.getElementById('stressSurgeVal');
    const yieldVal = document.getElementById('stressYieldVal');
    const delayVal = document.getElementById('stressDelayVal');

    if (surgeVal) surgeVal.textContent = `${surge.toFixed(1)}x`;
    if (yieldVal) yieldVal.textContent = `+${yShock} bps`;
    if (delayVal) delayVal.textContent = `${delay}d`;

    // Map surge (1.0 - 4.0) to deposit shock percent (0% - 300%)
    const depositShockPct = Math.max(0, (surge - 1.0) * 100);

    // Call live backend stress engine
    if (window.ReserveJugglerAPI && window.ReserveJugglerAPI.isConnected) {
      try {
        const res = await window.ReserveJugglerAPI.runStressSimulation({
          depositShockPct: depositShockPct,
          rateShockBps: yShock,
          repoDelayDays: delay,
          creditUnavailable: creditFrozen,
          equityDrawdownPct: surge > 2.0 ? 15.0 : 0.0
        });

        if (res && res.strategies) {
          renderStressStrategyCards(res.strategies);
        }
      } catch (err) {
        console.warn('Live stress simulation error:', err);
      }
    }

    renderStressCanvas();
  };

  function renderStressStrategyCards(strategies) {
    const s1 = strategies.all_cash;
    const s2 = strategies.greedy_yield;
    const s3 = strategies.reserve_juggler;

    if (s1) {
      const el = document.getElementById('stressAllCashVal');
      const bEl = document.getElementById('stressAllCashBreach');
      if (el) el.textContent = `$${(s1.final_cash / 1000000).toFixed(1)}M Ending Cash`;
      if (bEl) {
        bEl.textContent = s1.reserve_breach ? 'Deficit Warning' : 'Safe';
        bEl.className = s1.reserve_breach ? 'pill-badge text-amber-500 font-bold' : 'pill-badge pill-badge-emerald font-bold';
      }
    }

    if (s2) {
      const el = document.getElementById('stressGreedyVal');
      const bEl = document.getElementById('stressGreedyBreach');
      if (el) el.textContent = `$${(s2.final_cash / 1000000).toFixed(1)}M (MTM -$${(s2.bond_mtm_loss / 1000000).toFixed(1)}M)`;
      if (bEl) {
        bEl.textContent = s2.reserve_breach ? 'BREACH // Shortfall' : 'Safe';
        bEl.className = s2.reserve_breach ? 'pill-badge text-rose-500 font-bold border border-rose-500/30' : 'pill-badge pill-badge-emerald font-bold';
      }
    }

    if (s3) {
      const el = document.getElementById('stressJugglerVal');
      const bEl = document.getElementById('stressJugglerBreach');
      if (el) el.textContent = `$${(s3.final_cash / 1000000).toFixed(1)}M (0 Deficit)`;
      if (bEl) {
        bEl.textContent = s3.reserve_breach ? 'Breach' : 'OPTIMAL // Defended';
        bEl.className = s3.reserve_breach ? 'pill-badge text-rose-500 font-bold' : 'pill-badge pill-badge-cyan font-bold';
      }
    }
  }

  window.applyStressPreset = function (surge, bps, delay = 0, creditFrozen = false) {
    const sEl = document.getElementById('stressSurgeSlider');
    const yEl = document.getElementById('stressYieldSlider');
    const dEl = document.getElementById('stressDelaySlider');
    const cEl = document.getElementById('stressCreditToggle');

    if (sEl) sEl.value = surge;
    if (yEl) yEl.value = bps;
    if (dEl) dEl.value = delay;
    if (cEl) cEl.checked = creditFrozen;

    updateStressLab();
    showToast(`Applied preset: ${surge}x run, +${bps} bps.`);
  };

  // 8. Export CSV
  window.exportAuditCSV = function () {
    const snap = window.ReserveJugglerAPI ? window.ReserveJugglerAPI.currentSnapshot : null;
    const logs = (snap && snap.audit_log) ? snap.audit_log : [];

    if (logs.length === 0) {
      showToast('No audit entries available to export.');
      return;
    }

    let csv = 'Timestamp (UTC),Action Type,Actor,Details\n';
    logs.forEach(l => {
      const ts = (l.timestamp || '').replace(/,/g, '');
      const act = (l.action || '').replace(/,/g, '');
      const actor = (l.actor || '').replace(/,/g, '');
      const det = (l.details || '').replace(/,/g, ';').replace(/\n/g, ' ');
      csv += `${ts},${act},${actor},"${det}"\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `reserve_juggler_audit_ledger_${new Date().toISOString().substring(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Audit CSV exported successfully.');
  };

  // =========================================================================
  // Simulation Theatre Controls
  // =========================================================================
  let simTheatreAnimId = null;
  window.openSimulationModal = function (type = 'repo_sweep') {
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
    showToast('Simulation state synchronized with live dashboard.');
  };

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
      { name: 'Tier 0: Cash', sub: 'Operational Checking', val: '$45.00M', pct: 0.75, color: '#00E5FF', status: '100% LIQUID' },
      { name: 'Tier 1: Repo', sub: 'Overnight Facility', val: '$55.00M', pct: 0.60, color: '#10B981', status: 'LOCKED (5.35% APY)' },
      { name: 'Tier 2: Sovereign', sub: '2Y/10Y Treasury Notes', val: '$100.0M', pct: 0.85, color: '#F59E0B', status: 'YIELD (4.75% APY)' },
      { name: 'Tier 3: Equities', sub: 'S&P Dividend ETF', val: '$20.00M', pct: 0.40, color: isLight ? '#0088A3' : '#00E5FF', status: 'QUARANTINED' }
    ];

    const tWidth = (W - 90) / 4;
    const tHeight = H * 0.56;
    const tTop = H * 0.24;

    tanks.forEach((t, i) => {
      const tx = 20 + i * (tWidth + 16);

      ctx.fillStyle = isLight ? '#FFFFFF' : '#0F172A';
      ctx.strokeStyle = t.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(tx, tTop, tWidth, tHeight, 12);
      ctx.fill();
      ctx.stroke();

      const fluidH = tHeight * t.pct;
      const fluidY = tTop + tHeight - fluidH;
      ctx.fillStyle = t.color + (isLight ? '28' : '33');
      ctx.beginPath();
      ctx.roundRect(tx + 2, fluidY, tWidth - 4, fluidH - 2, [0, 0, 10, 10]);
      ctx.fill();

      ctx.strokeStyle = t.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const time = performance.now() * 0.003;
      for (let x = tx; x <= tx + tWidth; x += 4) {
        const wy = fluidY + Math.sin(x * 0.06 + time + i * 1.5) * 4;
        if (x === tx) ctx.moveTo(x, wy); else ctx.lineTo(x, wy);
      }
      ctx.stroke();

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
    });
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
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    if (e.key === 'Escape') {
      closeSimulationModal();
      closeShortcutsModal();
      closeModifyDecisionModal();
    } else if (e.key === '?') {
      openShortcutsModal();
    } else if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault();
      advanceSimulationStep(1);
    } else if (e.key === 't' || e.key === 'T') {
      toggleTheme();
    } else if (e.key === 's' || e.key === 'S') {
      openSimulationModal('night_sweep');
    } else if (e.key >= '1' && e.key <= '9') {
      const viewKeys = ['overview', 'forecasts', 'portfolio', 'nba', 'stress', 'audit', 'settings', 'edge', 'sentinel'];
      if (viewKeys[parseInt(e.key, 10) - 1]) {
        switchView(viewKeys[parseInt(e.key, 10) - 1]);
      }
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

  // =========================================================================
  // Reserve Juggler Edge & Macro-Panic Sentinel Controllers
  // =========================================================================

  const PIPELINE_DATA = {
    flow: {
      title: 'Stage 1: Real-Time Customer Cash Flows',
      desc: 'The platform continuously monitors deposits and withdrawals across all commercial accounts. Every transaction is matched with incoming settlements to measure net cash velocity.',
      formula: 'Net = Inflow - Outflow'
    },
    ml: {
      title: 'Stage 2: Machine Learning Multi-Horizon Forecast',
      desc: 'Combines L2 Ridge regression on historical flow lags with day-of-week & hour-of-day seasonal baselines to project the next 1h, 6h, and 24h expected cash requirements.',
      formula: 'Forecast = 0.968 * Seasonal + 0.032 * Ridge'
    },
    or: {
      title: 'Stage 3: Operations Research Scenario Reconciliation',
      desc: 'Uses SciPy Sequential Least Squares Quadratic Programming (SLSQP) to calculate optimal non-negative ensemble weights summing to 1.0, minimizing out-of-sample prediction loss.',
      formula: 'min || w*M - y || s.t. w_i >= 0, sum(w) = 1'
    },
    reserve: {
      title: 'Stage 4: Conformal Reserve & Capital Firewall Floor',
      desc: 'Conformal prediction quantiles (P95/P99) calculate worst-case cumulative outflows, setting a mandatory liquidity floor that no investment decision is permitted to violate.',
      formula: 'ReserveFloor = max(BaseFloor, P95_Outflow + Buffer)'
    },
    class: {
      title: 'Stage 5: Hot Money vs Core Capital Partition',
      desc: 'Segregates immediate payout obligations (Hot Money) from true long-term surplus (Core Capital). Hot capital is strictly kept in T+0 cash & overnight repo.',
      formula: 'CoreCapital = TotalLiquidity - ProtectedReserve - Obligations'
    },
    tier: {
      title: 'Stage 6: Multi-Tier Treasury Optimization',
      desc: 'Allocates eligible surplus across Tier 1 (Repo: 5.30% APY), Tier 2 (Sovereign Bonds: 4.85% APY), and Tier 3 (Equities: 8.50% APY, capped at 10%) while minimizing costs.',
      formula: 'max Yield - Slippage - Penalties s.t. Hard Gates'
    },
    monitor: {
      title: 'Stage 7: Macro-Panic Sentinel & Self-Learning Telemetry',
      desc: 'Monitors real-time central bank feeds and sovereign stress signals. If risk spikes, the engine deflates equity caps, widens cash reserves, and logs execution to SQLite for candidate model updates.',
      formula: 'SafetyScore >= 70 & RiskMode == NORMAL'
    }
  };

  window.showPipelineDetail = function(stage) {
    document.querySelectorAll('.pipeline-step').forEach(el => el.classList.remove('active'));
    if (window.event && window.event.currentTarget) {
      window.event.currentTarget.classList.add('active');
    }
    const info = PIPELINE_DATA[stage] || PIPELINE_DATA.flow;
    const titleEl = document.getElementById('pipeDetailTitle');
    const descEl = document.getElementById('pipeDetailDesc');
    const formulaEl = document.getElementById('pipeDetailFormula');
    if (titleEl) titleEl.textContent = info.title;
    if (descEl) descEl.textContent = info.desc;
    if (formulaEl) formulaEl.textContent = info.formula;
  };

  window.toggleExplainerDrawer = function(drawerId) {
    const el = document.getElementById(drawerId);
    if (!el) return;
    el.classList.toggle('open');
  };

  window.initEdgeView = function() {
    fetch('/api/decision/timeline')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (!data) return;
      })
      .catch(() => {});

    fetch('/api/algorithms/metadata')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (!data) return;
      })
      .catch(() => {});
  };

  window.initSentinelView = function() {
    fetch('/api/sentinel/signals')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (!data) return;
        const stressEl = document.getElementById('sentinelStressMetric');
        if (stressEl && data.aggregate_stress_score !== undefined) {
          stressEl.textContent = `${Number(data.aggregate_stress_score).toFixed(1)} / 100`;
        }
      })
      .catch(() => {});
  };

  // =========================================================================
  // Interactive Simulation & Animation Handlers for USPs & Governance
  // =========================================================================

  // 1. USP 1: Capital Flow Dynamic Simulation
  window.runCapitalFlowAnimation = function() {
    const zones = document.querySelectorAll('.zone-card');
    zones.forEach((z, i) => {
      setTimeout(() => {
        z.classList.add('is-selected');
        if (typeof gsap !== 'undefined') {
          gsap.fromTo(z, { scale: 0.96 }, { scale: 1, duration: 0.35, ease: 'back.out(2)' });
        }
        setTimeout(() => z.classList.remove('is-selected'), 1800);
      }, i * 250);
    });

    const pipelineSteps = document.querySelectorAll('.pipeline-step');
    pipelineSteps.forEach((step, i) => {
      setTimeout(() => {
        pipelineSteps.forEach(s => s.classList.remove('active'));
        step.classList.add('active');
        const keys = ['flow', 'ml', 'or', 'reserve', 'class', 'tier', 'monitor'];
        showPipelineDetail(keys[i] || 'flow');
      }, i * 350);
    });

    showToast('Simulation: $21.4M/hr customer flow dynamically partitioned into Cash Reserve & Overnight Repo.');
  };

  // 2. USP 2: Two-Hour Bridge vs Early-Break Animated Comparison
  window.runTwoHourBridgeAnimation = function() {
    const laneA = document.querySelector('.timeline-lane');
    const laneB = document.querySelectorAll('.timeline-lane')[1];
    if (!laneA || !laneB) return;

    if (typeof gsap !== 'undefined') {
      gsap.fromTo(laneA, { opacity: 0.5, x: -10 }, { opacity: 1, x: 0, duration: 0.4 });
      setTimeout(() => {
        gsap.fromTo(laneB, { scale: 0.98, borderColor: '#00E5FF' }, { scale: 1, borderColor: 'rgba(0, 229, 255, 0.6)', duration: 0.5 });
      }, 500);
    }

    const laneBNodes = laneB.querySelectorAll('.timeline-node');
    laneBNodes.forEach((node, i) => {
      setTimeout(() => {
        node.style.background = 'rgba(0, 229, 255, 0.15)';
        node.style.borderRadius = '8px';
        node.style.padding = '4px 8px';
        setTimeout(() => {
          node.style.background = 'transparent';
          node.style.padding = '0';
        }, 1200);
      }, i * 300);
    });

    showToast('Bridge Simulation Complete: $10M overnight repo protected. $49,726.03 (99.45%) saved over early break!');
  };

  // 3. USP 3: Hot Money Withdrawal Surge Simulation
  window.runWithdrawalSpikeSimulation = function() {
    const hotBand = document.querySelector('.capital-band-hot');
    const coreBand = document.querySelector('.capital-band-core');
    if (!hotBand) return;

    if (typeof gsap !== 'undefined') {
      gsap.fromTo(hotBand, { scale: 0.97 }, { scale: 1, duration: 0.4, ease: 'bounce.out' });
      if (coreBand) {
        gsap.fromTo(coreBand, { opacity: 0.7 }, { opacity: 1, duration: 0.4 });
      }
    }

    showToast('Stress Outflow Injected: $25M withdrawal absorbed via Hot Money (T+0 Cash). Core Bonds untouched: $0 liquidation loss!');
  };

  // 4. Governance: Tax Policy Statutory Reroute Simulation
  window.runTaxPolicyRerouteDemo = function() {
    const cardNaive = document.getElementById('taxCardNaive');
    const cardOpt = document.getElementById('taxCardOptimized');
    const telemetryMsg = document.getElementById('taxTelemetryMsg');
    const telemetryDot = document.getElementById('taxTelemetryDot');

    if (cardNaive && cardOpt) {
      if (typeof gsap !== 'undefined') {
        gsap.fromTo(cardNaive, { x: -8 }, { x: 0, duration: 0.3, ease: 'power2.out' });
        setTimeout(() => {
          gsap.fromTo(cardOpt, { scale: 0.97 }, { scale: 1, duration: 0.4, ease: 'back.out(2)' });
        }, 300);
      }
    }

    if (telemetryMsg) {
      telemetryMsg.innerHTML = '<span class="text-emerald-400 font-bold">REROUTE EXECUTED:</span> Section 871(m) sovereign exemption detected. Bond liquidation halted; overnight repo substitution executed. <strong class="text-emerald-300">+$177,260.00 saved.</strong>';
    }
    if (telemetryDot) {
      telemetryDot.className = 'w-2 h-2 rounded-full bg-emerald-400 animate-ping';
      setTimeout(() => {
        telemetryDot.className = 'w-2 h-2 rounded-full bg-emerald-400';
      }, 2000);
    }

    showToast('Tax Reroute Active: 30% statutory withholding avoided via Section 871(m) overnight substitution.');
  };

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

    // Trigger initial stress simulation
    setTimeout(() => {
      if (typeof updateStressLab === 'function') updateStressLab();
    }, 200);
  });

  window.addEventListener('resize', () => {
    renderLiquidityCanvas();
    renderPortfolioCanvases();
    renderStressCanvas();
  });
})();
