/**
 * =============================================================================
 * RESERVE*JUGGLER — UNIFIED FRONTEND API CLIENT & SYNCHRONIZER (api-client.js)
 * Connects FOSS UI directly to the FastAPI Backend ApplicationFacade.
 * Provides REST endpoints, SSE real-time streaming, and dashboard synchronization.
 * =============================================================================
 */

(function (window) {
  'use strict';

  class ReserveJugglerAPIClient {
    constructor() {
      // Auto-detect base URL:
      if (window.location.protocol.startsWith('http')) {
        this.baseUrl = window.location.origin;
      } else {
        this.baseUrl = 'http://localhost:8089';
      }

      this.isConnected = false;
      this.latencyMs = 0;
      this.currentSnapshot = null;
      this.eventSource = null;
      this.snapshotListeners = [];
      this.statusListeners = [];
      this.isPolling = false;
      this.pollInterval = null;
      this._candidateBaseUrls = [
        window.location.protocol.startsWith('http') ? window.location.origin : 'http://localhost:8089',
        'http://localhost:8089',
        'http://localhost:8000',
        'http://127.0.0.1:8089'
      ];

      // Auto-initialize connection check
      this.checkConnection();
    }

    setBaseUrl(url) {
      this.baseUrl = url.replace(/\/+$/, '');
      this.checkConnection();
    }

    onSnapshot(fn) {
      if (typeof fn === 'function') {
        this.snapshotListeners.push(fn);
        if (this.currentSnapshot) {
          try { fn(this.currentSnapshot); } catch (e) { console.error('Snapshot listener error:', e); }
        }
      }
    }

    onStatusChange(fn) {
      if (typeof fn === 'function') {
        this.statusListeners.push(fn);
        fn(this.isConnected, this.latencyMs);
      }
    }

    _notifyStatus(connected, latency) {
      this.isConnected = connected;
      this.latencyMs = latency;
      this.statusListeners.forEach(fn => {
        try { fn(connected, latency); } catch (e) { console.error(e); }
      });
      this._updateConnectionIndicatorUI();
    }

    _notifySnapshot(snap) {
      this.currentSnapshot = snap;
      this.syncDashboard(snap);
      this.snapshotListeners.forEach(fn => {
        try { fn(snap); } catch (e) { console.error(e); }
      });
    }

    async _request(path, options = {}) {
      const url = `${this.baseUrl}${path}`;
      const t0 = performance.now();
      try {
        const headers = Object.assign({
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }, options.headers || {});

        const res = await fetch(url, { ...options, headers });
        const latency = Math.round(performance.now() - t0);

        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`API HTTP ${res.status}: ${errText}`);
        }

        const data = await res.json();
        this._notifyStatus(true, latency);
        return data;
      } catch (err) {
        this._notifyStatus(false, 0);
        console.warn(`[ReserveJugglerAPI] Request failed for ${path}:`, err.message);
        throw err;
      }
    }

    async checkConnection() {
      // Try candidate URLs if needed
      for (const cand of this._candidateBaseUrls) {
        try {
          const t0 = performance.now();
          const res = await fetch(`${cand}/api/health`, { headers: { 'Accept': 'application/json' } });
          if (res.ok) {
            const health = await res.json();
            if (health && health.status === 'HEALTHY') {
              this.baseUrl = cand;
              this._notifyStatus(true, Math.round(performance.now() - t0));
              this.startEventStream();
              await this.fetchSnapshot();
              return true;
            }
          }
        } catch (e) {
          // Continue to next candidate
        }
      }

      this._notifyStatus(false, 0);
      this._scheduleReconnect();
      return false;
    }

    _scheduleReconnect() {
      if (!this._reconnectTimer) {
        this._reconnectTimer = setTimeout(() => {
          this._reconnectTimer = null;
          this.checkConnection();
        }, 3000);
      }
    }

    _updateConnectionIndicatorUI() {
      const dot = document.getElementById('backendStatusDot');
      const text = document.getElementById('backendStatusText');
      const ping = document.getElementById('backendStatusPing');

      if (dot && text) {
        if (this.isConnected) {
          dot.className = 'w-2 h-2 rounded-full bg-emerald-400 pulse-dot';
          text.textContent = 'Backend Active';
          if (ping) ping.textContent = `${this.latencyMs}ms`;
        } else {
          dot.className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse';
          text.textContent = 'Connecting...';
          if (ping) ping.textContent = '--';
        }
      }
    }

    // =========================================================================
    // SSE Live Streaming / Polling
    // =========================================================================

    startEventStream() {
      if (typeof EventSource === 'undefined') {
        this.startPolling(3500);
        return;
      }

      if (this.eventSource) {
        this.eventSource.close();
      }

      try {
        const streamUrl = `${this.baseUrl}/api/stream`;
        this.eventSource = new EventSource(streamUrl);

        this.eventSource.addEventListener('snapshot', (e) => {
          try {
            const snap = JSON.parse(e.data);
            this._notifySnapshot(snap);
          } catch (err) {
            console.error('Failed to parse SSE snapshot:', err);
          }
        });

        this.eventSource.onerror = () => {
          if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
          }
          this.startPolling(4000);
        };
      } catch (e) {
        this.startPolling(3500);
      }
    }

    startPolling(intervalMs = 3500) {
      if (this.isPolling) return;
      this.isPolling = true;
      this.pollInterval = setInterval(async () => {
        try {
          await this.fetchSnapshot();
        } catch (e) {
          // Ignore polling errors
        }
      }, intervalMs);
    }

    stopPolling() {
      if (this.pollInterval) {
        clearInterval(this.pollInterval);
        this.pollInterval = null;
      }
      this.isPolling = false;
    }

    // =========================================================================
    // Core API Methods
    // =========================================================================

    async fetchHealth() {
      return this._request('/api/health');
    }

    async fetchStatus() {
      return this._request('/api/status');
    }

    async fetchSnapshot(asOf = null) {
      const q = asOf ? `?as_of=${encodeURIComponent(asOf)}` : '';
      const snap = await this._request(`/api/dashboard/snapshot${q}`);
      this._notifySnapshot(snap);
      return snap;
    }

    async switchScenario(scenarioName) {
      const res = await this._request('/api/scenario/switch', {
        method: 'POST',
        body: JSON.stringify({ scenario: scenarioName })
      });
      if (res && res.snapshot) {
        this._notifySnapshot(res.snapshot);
      }
      return res;
    }

    async submitDecisionAction({
      actionType,
      actorName = 'Chief Treasurer',
      actorRole = 'HEAD_OF_TREASURY',
      statedReason = 'Authorized under mandatory statutory bounds.',
      modifiedAmount = null,
      modifiedAction = null,
      isLiveExecution = false
    }) {
      const payload = {
        action_type: actionType,
        actor_name: actorName,
        actor_role: actorRole,
        stated_reason: statedReason,
        modified_amount: modifiedAmount ? parseFloat(modifiedAmount) : null,
        modified_action: modifiedAction,
        is_live_execution: isLiveExecution
      };
      const res = await this._request('/api/decision/action', {
        method: 'POST',
        body: JSON.stringify(payload)
      });
      if (res && res.snapshot) {
        this._notifySnapshot(res.snapshot);
      }
      return res;
    }

    async runDecisionCycle() {
      const res = await this._request('/api/decision/cycle', {
        method: 'POST',
        body: JSON.stringify({})
      });
      if (res && res.snapshot) {
        this._notifySnapshot(res.snapshot);
      }
      return res;
    }

    async runStressSimulation({
      depositShockPct = 0.0,
      rateShockBps = 0.0,
      repoDelayDays = 0,
      creditUnavailable = false,
      equityDrawdownPct = 0.0
    }) {
      return this._request('/api/stress/simulate', {
        method: 'POST',
        body: JSON.stringify({
          deposit_shock_pct: depositShockPct,
          rate_shock_bps: rateShockBps,
          repo_delay_days: repoDelayDays,
          credit_unavailable: creditUnavailable,
          equity_drawdown_pct: equityDrawdownPct
        })
      });
    }

    async fetchStressPresets() {
      return this._request('/api/stress/presets');
    }

    async fetchLearningPerformance() {
      return this._request('/api/learning/performance');
    }

    async fetchCandidates() {
      return this._request('/api/learning/candidates');
    }

    async proposeCandidate(data) {
      return this._request('/api/learning/propose', {
        method: 'POST',
        body: JSON.stringify(data)
      });
    }

    async promoteCandidate(candidateId, actorRole = 'HEAD_OF_TREASURY') {
      return this._request('/api/learning/promote', {
        method: 'POST',
        body: JSON.stringify({ candidate_id: candidateId, actor_role: actorRole })
      });
    }

    async rollbackCandidate(candidateId, actorRole = 'HEAD_OF_TREASURY') {
      return this._request('/api/learning/rollback', {
        method: 'POST',
        body: JSON.stringify({ candidate_id: candidateId, actor_role: actorRole })
      });
    }

    async fetchAuditLogs() {
      return this._request('/api/audit/logs');
    }

    async fetchSentinelSignals() {
      return this._request('/api/sentinel/signals');
    }

    // =========================================================================
    // Comprehensive UI Dashboard Synchronization
    // =========================================================================

    syncDashboard(snap) {
      if (!snap) return;

      const state = snap.treasury_state || {};
      const dec = snap.latest_decision || {};
      const exp = snap.decision_explanation || {};
      const fb = snap.forecast_bundle || {};
      const sentinel = snap.sentinel_result || {};
      const perf = snap.system_performance || {};

      // 1. Header UTC Clock & Stepper Badge
      const clockEl = document.getElementById('simClockDisplay');
      if (clockEl && snap.as_of_timestamp) {
        const dt = new Date(snap.as_of_timestamp);
        if (!isNaN(dt.getTime())) {
          const y = dt.getUTCFullYear();
          const m = String(dt.getUTCMonth() + 1).padStart(2, '0');
          const d = String(dt.getUTCDate()).padStart(2, '0');
          const h = String(dt.getUTCHours()).padStart(2, '0');
          const min = String(dt.getUTCMinutes()).padStart(2, '0');
          clockEl.textContent = `${y}-${m}-${d} ${h}:${min}`;
        }
      }

      const badgeEl = document.getElementById('simStepBadge');
      if (badgeEl) {
        badgeEl.textContent = snap.active_scenario || 'NORMAL';
      }

      const scenSel = document.getElementById('scenarioSelector');
      if (scenSel && snap.active_scenario) {
        scenSel.value = snap.active_scenario;
      }

      // 2. Cipher Capsule
      const cipherWord = document.getElementById('fhenixCipherWord');
      const cipherTail = document.getElementById('fhenixCipherTail');
      if (cipherWord) {
        const tss = dec.treasury_safety_score ? dec.treasury_safety_score.toFixed(1) : '86.5';
        cipherWord.textContent = `TSS:${tss}`;
      }
      if (cipherTail) {
        cipherTail.textContent = sentinel.system_status || 'OK';
      }

      // 3. Command Overview 6 KPI Cards
      const cash = state.available_cash || state.operational_cash_usd || 45000000;
      const reserve = dec.reserve || state.protected_reserve_floor || 25000000;
      const surplus = dec.eligible_surplus || Math.max(0, cash - reserve);
      const tss = dec.treasury_safety_score || 86.5;

      const cashEl = document.getElementById('kpiAvailableCash');
      if (cashEl) cashEl.textContent = `$${Math.round(cash).toLocaleString()}`;

      const reserveEl = document.getElementById('kpiProtectedReserve');
      if (reserveEl) reserveEl.textContent = `$${Math.round(reserve).toLocaleString()}`;

      const surplusEl = document.getElementById('kpiSurplus');
      if (surplusEl) surplusEl.textContent = `$${Math.round(surplus).toLocaleString()}`;

      const safetyScoreEl = document.getElementById('kpiSafetyScore');
      if (safetyScoreEl) {
        safetyScoreEl.innerHTML = `${tss.toFixed(1)} <span class="text-sm secondary-text font-normal">/ 100</span>`;
      }

      const annualReturnEl = document.getElementById('kpiAnnualReturn');
      if (annualReturnEl && state.total_balance_sheet_assets_usd) {
        annualReturnEl.textContent = `$${Math.round(state.total_balance_sheet_assets_usd).toLocaleString()}`;
      }

      const stressEl = document.getElementById('kpiStressSurvival');
      if (stressEl) {
        if (snap.active_scenario === 'BANKING_STRESS') {
          stressEl.textContent = 'SVB Run Shield';
        } else if (snap.active_scenario === 'RATE_HIKE_SHOCK') {
          stressEl.textContent = 'Rate Shock Defended';
        } else {
          stressEl.textContent = '72h+ Horizon';
        }
      }

      // 4. Update Simulation Engine Memory (for smooth canvas animations)
      if (window.simEngine) {
        window.simEngine.availableCash = cash;
        window.simEngine.protectedReserve = reserve;
        window.simEngine.investableSurplus = surplus;
      }

      // 5. Quantile Forecasts View Table & Metrics
      this._syncForecastsView(fb);

      // 6. 4-Tier Portfolio View
      this._syncPortfolioView(state);

      // 7. Next Best Action (NBA) View & Explainability
      this._syncNbaView(dec, exp);

      // 8. Auditable Ledger View
      this._syncAuditView(snap.audit_log || []);

      // 9. Governance & Policy View
      this._syncGovernanceView(sentinel, perf, snap.active_candidates || []);
    }

    _syncForecastsView(fb) {
      if (!fb || !fb.point_forecasts) return;

      const points = fb.point_forecasts;
      const ptsList = Array.isArray(points) ? points : Object.values(points);

      // 24h forecast summary cards if available
      const p24 = ptsList.find(p => p.horizon === '24h' || p.horizon_hours === 24);
      if (p24) {
        const expIn = p24.deposit_point_forecast || p24.expected_inflow || 18000000;
        const expOut = p24.withdrawal_point_forecast || p24.expected_outflow || 32000000;
        const net = p24.net_outflow_p95 !== undefined ? p24.net_outflow_p95 : (expIn - expOut);

        const inEl = document.getElementById('fcExpectedInflow');
        if (inEl) inEl.textContent = `+$${Math.round(expIn).toLocaleString()}`;

        const outEl = document.getElementById('fcExpectedOutflow');
        if (outEl) outEl.textContent = `-$${Math.round(expOut).toLocaleString()}`;

        const netEl = document.getElementById('fcNetDelta');
        if (netEl) netEl.textContent = `${net >= 0 ? '+' : ''}$${Math.round(net).toLocaleString()}`;
      }

      // Populate Forecast Horizons Table
      const tbody = document.getElementById('forecastTableBody');
      if (tbody) {
        tbody.innerHTML = ptsList.map(pt => {
          const horiz = pt.horizon || (pt.horizon_hours ? `${pt.horizon_hours}h` : '--');
          const dep = pt.deposit_point_forecast || pt.expected_inflow || 0;
          const wdl = pt.withdrawal_point_forecast || pt.expected_outflow || 0;
          const p95 = pt.net_outflow_p95 || pt.uncertainty_p95 || 0;
          const rel = pt.reliability_tier || (pt.confidence_score ? `${(pt.confidence_score * 100).toFixed(0)}%` : 'HIGH');

          return `
            <tr class="border-b transition-colors hover:bg-white/5" style="border-color: var(--border-color);">
              <td class="py-2.5 font-bold font-mono" style="color: var(--cyan-primary);">${horiz}</td>
              <td class="py-2.5 font-mono text-emerald-500">+$${Math.round(dep).toLocaleString()}</td>
              <td class="py-2.5 font-mono text-rose-500">-$${Math.round(wdl).toLocaleString()}</td>
              <td class="py-2.5 font-mono text-amber-500">$${Math.round(p95).toLocaleString()}</td>
              <td class="py-2.5 font-mono"><span class="pill-badge pill-badge-cyan text-[10px] font-bold">${rel}</span></td>
            </tr>
          `;
        }).join('');
      }
    }

    _syncPortfolioView(state) {
      if (!state) return;

      const totalBs = state.total_balance_sheet_assets_usd || 265000000;
      const totalLiq = state.total_liquidatable_value_usd || 255150000;

      const bsEl = document.getElementById('portfolioTotalBalanceSheet');
      if (bsEl) bsEl.textContent = `$${Math.round(totalBs).toLocaleString()}`;

      const liqEl = document.getElementById('portfolioTotalLiquidatable');
      if (liqEl) liqEl.textContent = `$${Math.round(totalLiq).toLocaleString()}`;

      const positions = state.asset_positions || [];

      // Update Tiers summary cards
      const t0Cash = positions.find(p => p.asset_class === 'CASH')?.market_value || state.available_cash || 45000000;
      const t1Repo = positions.filter(p => p.asset_class === 'REPO').reduce((s, p) => s + p.market_value, 0) || 55000000;
      const t2Bonds = positions.filter(p => p.asset_class === 'SOVEREIGN_BONDS').reduce((s, p) => s + p.market_value, 0) || 100000000;
      const t3Eq = positions.filter(p => p.asset_class === 'EQUITIES').reduce((s, p) => s + p.market_value, 0) || 20000000;

      const t0El = document.getElementById('tier0Balance');
      if (t0El) t0El.textContent = `$${Math.round(t0Cash).toLocaleString()}`;

      const t1El = document.getElementById('tier1Balance');
      if (t1El) t1El.textContent = `$${Math.round(t1Repo).toLocaleString()}`;

      const t2El = document.getElementById('tier2Balance');
      if (t2El) t2El.textContent = `$${Math.round(t2Bonds).toLocaleString()}`;

      const t3El = document.getElementById('tier3Balance');
      if (t3El) t3El.textContent = `$${Math.round(t3Eq).toLocaleString()}`;

      // Populate positions table
      const tbody = document.getElementById('portfolioPositionsTableBody');
      if (tbody && positions.length > 0) {
        tbody.innerHTML = positions.map(pos => {
          return `
            <tr class="border-b transition-colors hover:bg-white/5" style="border-color: var(--border-color);">
              <td class="py-2.5 font-bold font-mono text-xs" style="color: var(--cyan-primary);">${pos.asset_id}</td>
              <td class="py-2.5 text-xs font-medium heading-text">${pos.name}</td>
              <td class="py-2.5 font-mono text-xs"><span class="pill-badge text-[10px] font-bold">${pos.asset_class}</span></td>
              <td class="py-2.5 font-mono text-xs heading-text font-bold">$${Math.round(pos.market_value).toLocaleString()}</td>
              <td class="py-2.5 font-mono text-xs text-emerald-500 font-bold">$${Math.round(pos.accessible_liquidity).toLocaleString()}</td>
              <td class="py-2.5 font-mono text-xs text-amber-500">${pos.yield_pct ? `${pos.yield_pct.toFixed(2)}%` : '--'}</td>
              <td class="py-2.5 font-mono text-xs muted-text">${pos.settlement_days !== undefined ? `T+${pos.settlement_days}` : '--'}</td>
            </tr>
          `;
        }).join('');
      }
    }

    _syncNbaView(dec, exp) {
      if (!dec) return;

      const action = dec.selected_action || 'DRAW_REPO_30D';
      const asset = dec.target_asset || 'REPO-30D-GCF';
      const amount = dec.amount || 15000000;
      const status = dec.approval_status || 'PENDING_APPROVAL';
      const costs = dec.estimated_costs || {};
      const totalCost = costs.total !== undefined ? costs.total : 8575.0;

      const titleEl = document.getElementById('nbaRecommendationTitle');
      if (titleEl) titleEl.textContent = `${action} &bull; ${asset}`;

      const descEl = document.getElementById('nbaRecommendationDesc');
      if (descEl) descEl.textContent = `Recommended optimal liquidity draw of $${Math.round(amount).toLocaleString()} targeting ${asset} at minimal all-in cost.`;

      const badge = document.getElementById('nbaStatusBadge');
      if (badge) {
        badge.textContent = status;
        if (status === 'APPROVED' || status === 'EXECUTED') {
          badge.className = 'pill-badge pill-badge-emerald font-bold';
        } else if (status === 'REJECTED' || status === 'CANCELLED') {
          badge.className = 'pill-badge text-rose-400 border border-rose-500/30 font-bold';
        } else {
          badge.className = 'pill-badge pill-badge-cyan font-bold';
        }
      }

      const allocEl = document.getElementById('nbaOptimalAllocation');
      if (allocEl) allocEl.textContent = `$${Math.round(amount).toLocaleString()}`;

      const costEl = document.getElementById('nbaEstimatedCost');
      if (costEl) costEl.textContent = `$${totalCost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

      // Hard Gates Table
      const gates = (exp && exp.hard_gates) ? exp.hard_gates : (dec.hard_gate_outcomes || []);
      const gatesTbody = document.getElementById('hardGatesTableBody');
      if (gatesTbody && gates.length > 0) {
        gatesTbody.innerHTML = gates.map(g => {
          const passed = g.passed !== undefined ? g.passed : (g.status === 'PASS');
          const statusBadge = passed
            ? '<span class="pill-badge pill-badge-emerald text-[10px] font-bold">PASS</span>'
            : '<span class="pill-badge text-rose-400 border border-rose-500/30 text-[10px] font-bold">FAIL</span>';

          return `
            <tr class="border-b transition-colors hover:bg-white/5" style="border-color: var(--border-color);">
              <td class="py-2.5 font-bold font-mono text-xs">${g.gate_name}</td>
              <td class="py-2.5">${statusBadge}</td>
              <td class="py-2.5 text-xs secondary-text">${g.reason || 'Safety threshold evaluated'}</td>
            </tr>
          `;
        }).join('');
      }

      // Rejected Alternatives
      const alts = (exp && exp.rejected_alternatives) ? exp.rejected_alternatives : (dec.alternatives || []);
      const altsContainer = document.getElementById('rejectedAlternativesContainer');
      if (altsContainer && alts.length > 0) {
        altsContainer.innerHTML = alts.map(alt => {
          return `
            <div class="curved-card p-4 space-y-1.5 border" style="border-color: var(--border-color);">
              <div class="flex items-center justify-between">
                <span class="font-bold font-mono text-xs text-rose-400">${alt.action_type || alt.target_asset}</span>
                <span class="pill-badge text-[9px] font-bold text-rose-400 border border-rose-500/30">REJECTED</span>
              </div>
              <p class="text-[11px] secondary-text">${alt.rejection_reason || 'Suboptimal cost or latency mismatch.'}</p>
            </div>
          `;
        }).join('');
      }
    }

    _syncAuditView(auditLog) {
      const tbody = document.getElementById('auditTableBody');
      if (!tbody || !auditLog || auditLog.length === 0) return;

      tbody.innerHTML = auditLog.slice().reverse().map(item => {
        const ts = item.timestamp ? item.timestamp.replace('T', ' ').substring(0, 19) : '--';
        const act = item.action || 'SYSTEM_EVENT';
        const actor = item.actor || 'System';
        const details = item.details || '';

        return `
          <tr class="border-b transition-colors hover:bg-white/5" style="border-color: var(--border-color);">
            <td class="py-2.5 font-mono text-xs muted-text">${ts}</td>
            <td class="py-2.5 font-bold font-mono text-xs" style="color: var(--cyan-primary);">${act}</td>
            <td class="py-2.5 text-xs font-mono secondary-text">${actor}</td>
            <td class="py-2.5 text-xs secondary-text">${details}</td>
          </tr>
        `;
      }).join('');
    }

    _syncGovernanceView(sentinel, perf, candidates) {
      // Sentinel Score & Status
      const sentStatusEl = document.getElementById('govSentinelStatus');
      if (sentStatusEl) sentStatusEl.textContent = sentinel.system_status || 'OK';

      const sentScoreEl = document.getElementById('govMacroStressScore');
      if (sentScoreEl) sentScoreEl.textContent = `${(sentinel.aggregate_macro_stress_score || 0).toFixed(1)} / 100`;

      // Active Signals
      const sigs = sentinel.active_signals || [];
      const sigsTbody = document.getElementById('govSignalsTableBody');
      if (sigsTbody && sigs.length > 0) {
        sigsTbody.innerHTML = sigs.map(sig => {
          return `
            <tr class="border-b transition-colors hover:bg-white/5" style="border-color: var(--border-color);">
              <td class="py-2 font-mono text-xs font-bold" style="color: var(--cyan-primary);">${sig.signal_id}</td>
              <td class="py-2 text-xs font-mono secondary-text">${sig.category}</td>
              <td class="py-2"><span class="pill-badge pill-badge-cyan text-[10px] font-bold">${sig.severity}</span></td>
              <td class="py-2 font-mono text-xs ${sig.sentiment_score < 0 ? 'text-rose-500' : 'text-emerald-500'}">${sig.sentiment_score >= 0 ? '+' : ''}${sig.sentiment_score.toFixed(2)}</td>
            </tr>
          `;
        }).join('');
      }

      // Self-Learning Stats
      const totalEventsEl = document.getElementById('govTotalEvents');
      if (totalEventsEl) totalEventsEl.textContent = (perf.total_learning_events || 0).toLocaleString();

      const matchedPairsEl = document.getElementById('govMatchedPairs');
      if (matchedPairsEl) matchedPairsEl.textContent = perf.matched_decision_outcome_pairs || 0;

      const gatePassRateEl = document.getElementById('govGatePassRate');
      if (gatePassRateEl && perf.decision_metrics) {
        gatePassRateEl.textContent = `${((perf.decision_metrics.hard_gate_pass_rate || 1.0) * 100).toFixed(1)}%`;
      }

      // Active Candidates Table
      const candTbody = document.getElementById('govCandidatesTableBody');
      if (candTbody && candidates.length > 0) {
        candTbody.innerHTML = candidates.map(c => {
          return `
            <tr class="border-b transition-colors hover:bg-white/5" style="border-color: var(--border-color);">
              <td class="py-2 font-mono text-xs font-bold" style="color: var(--cyan-primary);">${c.candidate_id}</td>
              <td class="py-2 text-xs font-mono">${c.component} // ${c.parameter_key}</td>
              <td class="py-2"><span class="pill-badge text-[10px] font-bold text-amber-400 border border-amber-500/30">${c.status || 'SHADOW_EVAL'}</span></td>
              <td class="py-2 text-right">
                <button onclick="window.promoteCandidateAction('${c.candidate_id}')" class="curved-btn py-1 px-2.5 text-[10px] font-bold font-mono text-emerald-400 hover:border-emerald-500">
                  Promote
                </button>
              </td>
            </tr>
          `;
        }).join('');
      }
    }
  }

  // Instantiate singleton on window
  window.ReserveJugglerAPI = new ReserveJugglerAPIClient();

})(window);
