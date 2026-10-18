/**
 * =========================================================
 * RESERVE*JUGGLER — INSTITUTIONAL SIMULATIONS ENGINE
 * Monte Carlo Forecasting, Repo Sweep Theatre & Stress Lab
 * =========================================================
 */

(function () {
  'use strict';

  class TreasurySimulationEngine {
    constructor() {
      this.currentHorizon = 24;
      this.simHourIndex = 5920;
      this.simDate = new Date(Date.UTC(2026, 5, 15, 16, 0, 0));
      this.availableCash = 24810500;
      this.protectedReserve = 18200000;
      this.investableSurplus = 6610500;
      this.seriesData = [];
      this.autoPlayInterval = null;

      // GSAP Interpolation variables
      this.currentPlotPoints = [];
      this.targetPlotPoints = [];
      this.graphAnimTween = { progress: 1.0 };

      // Status Cipher States
      this.statusStates = [
        { word: 'yield:4.89%', tail: 'OK' },
        { word: 'vaR_cov:99.5%', tail: 'LK' },
        { word: 'bridge:$274', tail: 'GO' },
        { word: 'svb_shield:ON', tail: '94' }
      ];
      this.statusIdx = 0;
    }

    init() {
      this.generateTelemetryData(72);
      this.targetPlotPoints = this.seriesData
        .filter(d => d.stepOffset <= this.currentHorizon)
        .map(p => p.cash);
      this.currentPlotPoints = [...this.targetPlotPoints];
    }

    /**
     * Generates continuous two-sided stochastic cash flow data
     */
    generateTelemetryData(hours = 72) {
      this.seriesData = [];
      for (let i = -12; i <= hours; i++) {
        const isForecast = i > 0;
        const hourOfDay = (16 + i + 2400) % 24;
        const wave = Math.sin((i + this.simHourIndex * 0.18) * 0.38) * 2900000;
        const cash = Math.max(18800000, Math.min(34500000, 25000000 + wave));
        const uncertainty = isForecast ? Math.min(4800000, 600000 + i * 90000) : 0;

        this.seriesData.push({
          stepOffset: i,
          hourLabel: `${String(hourOfDay).padStart(2, '0')}:00`,
          isForecast,
          cash,
          upper: cash + uncertainty,
          lower: Math.max(15800000, cash - uncertainty),
          deposits: 19200000 + Math.round(Math.sin(i * 0.5) * 4400000),
          withdrawals: 18100000 + Math.round(Math.cos(i * 0.45) * 4100000),
          eventTag: hourOfDay === 17 ? '17:00 Sweep Cutoff' : (hourOfDay === 8 ? '08:00 Repo Maturity' : null)
        });
      }
    }

    /**
     * Advances simulation forward with smooth GSAP animations
     */
    advanceStep(hours = 1) {
      this.simHourIndex += hours;
      this.simDate.setUTCHours(this.simDate.getUTCHours() + hours);

      // Fluctuate cash balance based on step
      const delta = (Math.sin(this.simHourIndex * 0.4) * 850000) + (Math.random() - 0.5) * 200000;
      const oldCash = this.availableCash;
      this.availableCash = Math.max(19000000, Math.min(36000000, this.availableCash + delta));

      // Update Header Clock
      const clockEl = document.getElementById('simClockDisplay');
      if (clockEl) {
        const y = this.simDate.getUTCFullYear();
        const m = String(this.simDate.getUTCMonth() + 1).padStart(2, '0');
        const d = String(this.simDate.getUTCDate()).padStart(2, '0');
        const h = String(this.simDate.getUTCHours()).padStart(2, '0');
        clockEl.textContent = `${y}-${m}-${d} ${h}:00`;
      }

      const badgeEl = document.getElementById('simStepBadge');
      if (badgeEl) {
        badgeEl.textContent = `H-${this.simHourIndex}`;
      }

      // Rolling GSAP Dollar values
      this.animateCurrencyValue('kpiAvailableCash', oldCash, this.availableCash);

      // Recalculate and trigger chart curve morphing
      this.generateTelemetryData(72);
      this.animateChartToNewData();

      // Trigger status cycle
      this.triggerStatusCycle();
    }

    animateCurrencyValue(elementId, startVal, endVal, prefix = '$') {
      const el = document.getElementById(elementId);
      if (!el) return;
      if (typeof gsap === 'undefined') {
        el.textContent = prefix + Math.round(endVal).toLocaleString('en-US');
        return;
      }
      gsap.to({ val: startVal }, {
        val: endVal,
        duration: 0.85,
        ease: 'power3.out',
        onUpdate: function () {
          el.textContent = prefix + Math.round(this.targets()[0].val).toLocaleString('en-US');
        }
      });
    }

    /**
     * Morph graph spline smoothly via GSAP
     */
    animateChartToNewData() {
      const visible = this.seriesData.filter(d => d.stepOffset <= this.currentHorizon);
      const newTargets = visible.map(d => d.cash);
      const startPoints = this.currentPlotPoints.length === newTargets.length ? [...this.currentPlotPoints] : null;

      this.targetPlotPoints = newTargets;
      if (!startPoints) {
        this.currentPlotPoints = [...this.targetPlotPoints];
        if (window.renderLiquidityCanvas) window.renderLiquidityCanvas();
        return;
      }

      this.graphAnimTween.progress = 0;
      if (typeof gsap !== 'undefined') {
        gsap.to(this.graphAnimTween, {
          progress: 1.0,
          duration: 0.85,
          ease: 'power2.out',
          onUpdate: () => {
            this.currentPlotPoints = this.targetPlotPoints.map((target, idx) => {
              const start = startPoints[idx] !== undefined ? startPoints[idx] : target;
              return start + (target - start) * this.graphAnimTween.progress;
            });
            if (window.renderLiquidityCanvas) window.renderLiquidityCanvas();
          }
        });
      } else {
        this.currentPlotPoints = [...this.targetPlotPoints];
        if (window.renderLiquidityCanvas) window.renderLiquidityCanvas();
      }
    }

    triggerStatusCycle() {
      const el = document.getElementById('fhenixCipherWord');
      const tail = document.getElementById('fhenixCipherTail');
      if (!el || typeof gsap === 'undefined') return;
      this.statusIdx = (this.statusIdx + 1) % this.statusStates.length;
      const next = this.statusStates[this.statusIdx];
      gsap.fromTo([el, tail],
        { y: -6, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.28,
          ease: 'power3.out',
          onStart: () => {
            el.textContent = next.word;
            tail.textContent = next.tail;
          }
        }
      );
    }
  }

  window.TreasurySimulationEngine = TreasurySimulationEngine;
})();
