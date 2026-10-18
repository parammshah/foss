/**
 * =========================================================
 * RESERVE*JUGGLER — CURRENCY STREAM SIMULATION ENGINE
 * Continuous 3D Flowing Banknotes (USD, EUR, GBP, CYBER)
 * High-Performance Procedural Canvas Architecture
 * =========================================================
 */

(function () {
  'use strict';

  class CurrencyStreamEngine {
    constructor(canvasId, options = {}) {
      this.canvas = document.getElementById(canvasId);
      if (!this.canvas) return;

      this.ctx = this.canvas.getContext('2d', { alpha: true });
      this.options = Object.assign({
        noteCount: 88,
        speed: 1.25,
        windAngle: 0.38, // Radians (~22 degrees diagonal drift)
        enableMouseWake: true,
        interactiveWind: true,
        ambientGlow: true
      }, options);

      this.notes = [];
      this.billCanvases = {};
      this.width = window.innerWidth;
      this.height = window.innerHeight;
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);

      this.mouse = {
        x: -1000,
        y: -1000,
        vx: 0,
        vy: 0,
        lastX: -1000,
        lastY: -1000,
        active: false
      };

      this.time = 0;
      this.isRunning = false;

      this.init();
    }

    init() {
      this.resize();
      this.generateBillTextures();
      this.spawnNotes();
      this.bindEvents();
      this.start();
    }

    resize() {
      this.width = window.innerWidth;
      this.height = window.innerHeight;
      this.canvas.width = this.width * this.dpr;
      this.canvas.height = this.height * this.dpr;
      this.canvas.style.width = `${this.width}px`;
      this.canvas.style.height = `${this.height}px`;
      this.ctx.scale(this.dpr, this.dpr);
    }

    /**
     * Procedurally pre-renders ultra-detailed banknote textures
     * into off-screen canvases for optimal 60+ FPS performance
     */
    generateBillTextures() {
      const types = ['usd', 'eur', 'gbp', 'cyber'];
      const W = 220; // Texture width
      const H = 100; // Texture height

      types.forEach(type => {
        const offCanvas = document.createElement('canvas');
        offCanvas.width = W;
        offCanvas.height = H;
        const octx = offCanvas.getContext('2d');

        // Crisp crisp edges
        octx.imageSmoothingEnabled = true;

        switch (type) {
          case 'usd':
            this.renderUSDTexture(octx, W, H);
            break;
          case 'eur':
            this.renderEURTexture(octx, W, H);
            break;
          case 'gbp':
            this.renderGBPTexture(octx, W, H);
            break;
          case 'cyber':
            this.renderCyberTexture(octx, W, H);
            break;
        }

        this.billCanvases[type] = offCanvas;
      });
    }

    /**
     * US $100 Federal Reserve Note ("Benjamin")
     */
    renderUSDTexture(ctx, w, h) {
      // Paper base: subtle green-cream
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, '#E6EDE5');
      grad.addColorStop(0.5, '#F0F5EC');
      grad.addColorStop(1, '#DEE8DD');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // Intricate guilloche border
      ctx.strokeStyle = '#2D5A3A';
      ctx.lineWidth = 3;
      ctx.strokeRect(3, 3, w - 6, h - 6);

      ctx.strokeStyle = '#3E7D52';
      ctx.lineWidth = 1;
      ctx.strokeRect(6, 6, w - 12, h - 12);

      // Lathe-work inner corner flourishes
      ctx.fillStyle = '#1B4D2E';
      ctx.font = 'bold 15px "IBM Plex Mono", monospace';
      ctx.fillText('100', 10, 22);
      ctx.fillText('100', w - 38, 22);
      ctx.fillText('100', 10, h - 10);
      ctx.fillText('100', w - 38, h - 10);

      // Central Oval Frame for Franklin
      ctx.strokeStyle = '#1F472B';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(w * 0.44, h * 0.5, 26, 36, 0, 0, Math.PI * 2);
      ctx.stroke();

      // Franklin silhouette graphic
      ctx.fillStyle = '#2C5539';
      ctx.beginPath();
      ctx.ellipse(w * 0.44, h * 0.46, 12, 16, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(w * 0.44, h * 0.72, 20, 14, 0, Math.PI, Math.PI * 2);
      ctx.fill();

      // Iconic 3D Blue Security Ribbon
      const ribbon = ctx.createLinearGradient(w * 0.62, 0, w * 0.68, 0);
      ribbon.addColorStop(0, '#1E40AF');
      ribbon.addColorStop(0.5, '#60A5FA');
      ribbon.addColorStop(1, '#1E3A8A');
      ctx.fillStyle = ribbon;
      ctx.fillRect(w * 0.62, 4, 14, h - 8);

      // Gold Treasury Seal
      ctx.fillStyle = '#B45309';
      ctx.beginPath();
      ctx.arc(w * 0.78, h * 0.5, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#FEF3C7';
      ctx.font = 'bold 8px monospace';
      ctx.fillText('USA', w * 0.73, h * 0.53);

      // Header Typography
      ctx.fillStyle = '#143820';
      ctx.font = 'bold 7px "Inter", sans-serif';
      ctx.fillText('FEDERAL RESERVE NOTE', w * 0.28, 16);
      ctx.font = '6px "IBM Plex Mono", monospace';
      ctx.fillText('THE UNITED STATES OF AMERICA', w * 0.24, 25);
    }

    /**
     * €100 Euro Note (Emerald Green Modern Architectural Arch)
     */
    renderEURTexture(ctx, w, h) {
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, '#E4F4EC');
      grad.addColorStop(0.5, '#ECF8F1');
      grad.addColorStop(1, '#D8EFE2');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // European Stars Circle (12 stars)
      ctx.fillStyle = '#059669';
      const cx = 30, cy = 34, r = 14;
      for (let i = 0; i < 12; i++) {
        const a = (i / 12) * Math.PI * 2;
        ctx.fillRect(cx + Math.cos(a) * r - 1.5, cy + Math.sin(a) * r - 1.5, 3, 3);
      }

      // Large 100 Numerals
      ctx.fillStyle = '#065F46';
      ctx.font = 'bold 22px "Inter", sans-serif';
      ctx.fillText('100', 12, h - 14);
      ctx.fillText('100', w - 48, 26);

      // Architectural Arch Bridge Motif
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(w * 0.55, h * 0.65, 28, Math.PI, 0);
      ctx.stroke();

      // Columns
      ctx.strokeRect(w * 0.55 - 24, h * 0.65, 8, 22);
      ctx.strokeRect(w * 0.55 + 16, h * 0.65, 8, 22);

      // Holographic Security Patch on Right
      const holo = ctx.createLinearGradient(w - 28, 0, w - 8, h);
      holo.addColorStop(0, '#34D399');
      holo.addColorStop(0.3, '#6EE7B7');
      holo.addColorStop(0.7, '#A7F3D0');
      holo.addColorStop(1, '#059669');
      ctx.fillStyle = holo;
      ctx.fillRect(w - 24, 6, 18, h - 12);

      // Euro Symbol
      ctx.fillStyle = '#064E3B';
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText('€', w - 20, h * 0.55);

      // European Central Bank typography
      ctx.font = '6px "IBM Plex Mono", monospace';
      ctx.fillText('BCE ECB EЦБ EZB EKP 2026', 56, 18);
    }

    /**
     * £50 British Banknote (Crimson-Red / Holographic Gold)
     */
    renderGBPTexture(ctx, w, h) {
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, '#FAECEF');
      grad.addColorStop(0.5, '#FFF5F7');
      grad.addColorStop(1, '#F3DEE3');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // Red filigree border
      ctx.strokeStyle = '#9F1239';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(4, 4, w - 8, h - 8);

      // Denomination
      ctx.fillStyle = '#881337';
      ctx.font = 'bold 20px "Inter", sans-serif';
      ctx.fillText('£50', 14, 28);
      ctx.fillText('£50', w - 46, h - 14);

      // Bank of England Header
      ctx.font = 'bold 7px "IBM Plex Mono", monospace';
      ctx.fillText('BANK OF ENGLAND', w * 0.35, 18);
      ctx.font = '6px sans-serif';
      ctx.fillText('I PROMISE TO PAY THE BEARER ON DEMAND', w * 0.3, 27);

      // Central Turing / Algorithmic Matrix Lattice
      ctx.strokeStyle = '#BE123C';
      ctx.lineWidth = 0.8;
      for (let x = w * 0.4; x < w * 0.7; x += 6) {
        ctx.beginPath();
        ctx.moveTo(x, 34);
        ctx.lineTo(x + 10, h - 14);
        ctx.stroke();
      }

      // Large Gold Metallic Foil Window
      const gold = ctx.createLinearGradient(20, h * 0.4, 45, h * 0.85);
      gold.addColorStop(0, '#D97706');
      gold.addColorStop(0.5, '#FDE68A');
      gold.addColorStop(1, '#92400E');
      ctx.fillStyle = gold;
      ctx.beginPath();
      ctx.ellipse(32, h * 0.62, 14, 18, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    /**
     * Autonomous Cyber Treasury Bill (Radiant Cyan / Obsidian Hologram)
     */
    renderCyberTexture(ctx, w, h) {
      // Obsidian gradient
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, '#090D18');
      grad.addColorStop(0.5, '#0E1726');
      grad.addColorStop(1, '#080C14');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // Cyan circuit frame
      ctx.strokeStyle = '#00E5FF';
      ctx.lineWidth = 2;
      ctx.strokeRect(4, 4, w - 8, h - 8);

      // Circuit node traces
      ctx.strokeStyle = 'rgba(0, 229, 255, 0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(10, 20); ctx.lineTo(40, 20); ctx.lineTo(55, 35);
      ctx.moveTo(w - 10, h - 20); ctx.lineTo(w - 40, h - 20); ctx.lineTo(w - 55, h - 35);
      ctx.stroke();

      // Brand Typography
      ctx.fillStyle = '#00E5FF';
      ctx.font = 'bold 12px "IBM Plex Mono", monospace';
      ctx.fillText('(reserve*juggler)', 14, 24);

      // High-denomination symbol
      ctx.font = 'extrabold 18px "Inter", sans-serif';
      ctx.fillText('10,000,000 ❖', 14, h - 14);

      // Central Conformal VaR cryptographic seal
      ctx.fillStyle = 'rgba(0, 229, 255, 0.15)';
      ctx.beginPath();
      ctx.arc(w * 0.6, h * 0.5, 24, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#00E5FF';
      ctx.stroke();

      ctx.fillStyle = '#38BDF8';
      ctx.font = 'bold 8px "IBM Plex Mono", monospace';
      ctx.fillText('VaR 99.5%', w * 0.52, h * 0.53);
    }

    /**
     * Spawns flowing currency notes distributed naturally across 3D depth
     */
    spawnNotes() {
      this.notes = [];
      const types = ['usd', 'eur', 'gbp', 'cyber'];

      for (let i = 0; i < this.options.noteCount; i++) {
        const type = types[i % types.length];
        this.notes.push(this.createNote(type, true));
      }
    }

    createNote(type, initialScatter = false) {
      // 3D coordinates & depth
      const z = 200 + Math.random() * 700; // 200 (close) to 900 (deep in background)
      const scale = 550 / (550 + z);

      let x, y;
      if (initialScatter) {
        x = Math.random() * (this.width + 200) - 100;
        y = Math.random() * (this.height + 200) - 100;
      } else {
        // Recycle entrance from top-left or left margin
        if (Math.random() < 0.6) {
          x = -150 - Math.random() * 200;
          y = Math.random() * this.height * 0.85;
        } else {
          x = Math.random() * this.width * 0.8;
          y = -150 - Math.random() * 200;
        }
      }

      // Stream velocity vectors
      const baseSpeed = (1.2 + Math.random() * 1.8) * this.options.speed * (0.6 + scale * 0.6);
      const angle = this.options.windAngle + (Math.random() - 0.5) * 0.28;

      return {
        type,
        x,
        y,
        z,
        scale,
        baseScale: scale,
        vx: Math.cos(angle) * baseSpeed,
        vy: Math.sin(angle) * baseSpeed,
        vz: (Math.random() - 0.5) * 0.4,
        
        // 3D Rotation & Aerodynamic Flutter
        pitch: Math.random() * Math.PI * 2,
        yaw: Math.random() * Math.PI * 2,
        roll: Math.random() * Math.PI * 2,
        vPitch: (Math.random() - 0.5) * 0.025,
        vYaw: (Math.random() - 0.5) * 0.035,
        vRoll: (Math.random() - 0.5) * 0.015,
        
        // Flutter wave phase
        flutterPhase: Math.random() * Math.PI * 2,
        flutterSpeed: 0.04 + Math.random() * 0.05,
        flutterAmplitude: 1.2 + Math.random() * 2.5,

        // Dimensions
        width: 140 * scale,
        height: 64 * scale,
        opacity: Math.min(0.85, 0.25 + scale * 0.55),
        isPerturbed: false
      };
    }

    bindEvents() {
      window.addEventListener('resize', () => {
        this.resize();
      });

      if (this.options.enableMouseWake) {
        window.addEventListener('mousemove', (e) => {
          if (this.mouse.lastX === -1000) {
            this.mouse.lastX = e.clientX;
            this.mouse.lastY = e.clientY;
          }
          this.mouse.vx = (e.clientX - this.mouse.lastX) * 0.3;
          this.mouse.vy = (e.clientY - this.mouse.lastY) * 0.3;
          this.mouse.x = e.clientX;
          this.mouse.y = e.clientY;
          this.mouse.lastX = e.clientX;
          this.mouse.lastY = e.clientY;
          this.mouse.active = true;

          clearTimeout(this.mouseTimer);
          this.mouseTimer = setTimeout(() => {
            this.mouse.vx *= 0.5;
            this.mouse.vy *= 0.5;
          }, 80);
        });

        window.addEventListener('mouseleave', () => {
          this.mouse.active = false;
        });
      }
    }

    start() {
      if (this.isRunning) return;
      this.isRunning = true;
      const loop = () => {
        if (!this.isRunning) return;
        this.update();
        this.render();
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }

    stop() {
      this.isRunning = false;
    }

    update() {
      this.time += 0.016;
      const boundaryRight = this.width + 180;
      const boundaryBottom = this.height + 180;

      for (let i = 0; i < this.notes.length; i++) {
        const n = this.notes[i];

        // Aerodynamic flutter oscillation
        n.flutterPhase += n.flutterSpeed;
        const flutterForce = Math.sin(n.flutterPhase) * n.flutterAmplitude;

        // Position progression
        n.x += n.vx + Math.cos(n.roll) * flutterForce * 0.3;
        n.y += n.vy + Math.sin(n.pitch) * flutterForce;
        n.z += n.vz;

        // Keep z in reasonable bounds
        if (n.z < 150) n.vz = Math.abs(n.vz);
        if (n.z > 950) n.vz = -Math.abs(n.vz);
        n.scale = 550 / (550 + n.z);
        n.width = 140 * n.scale;
        n.height = 64 * n.scale;

        // Rotation progression
        n.pitch += n.vPitch;
        n.yaw += n.vYaw;
        n.roll += n.vRoll;

        // Mouse Wake Interaction (Gentle gust of wind around cursor)
        if (this.mouse.active) {
          const dx = n.x - this.mouse.x;
          const dy = n.y - this.mouse.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const maxDist = 220;

          if (dist < maxDist && dist > 1) {
            const force = (1 - dist / maxDist) * 3.5;
            n.x += (dx / dist) * force * 2;
            n.y += (dy / dist) * force * 2;
            n.pitch += 0.06 * force;
            n.yaw += 0.08 * force;
          }
        }

        // Boundary Recycling: Smooth continuous inflow & outflow
        if (n.x > boundaryRight || n.y > boundaryBottom || n.x < -260 || n.y < -260) {
          const recycled = this.createNote(n.type, false);
          Object.assign(n, recycled);
        }
      }
    }

    render() {
      this.ctx.clearRect(0, 0, this.width, this.height);

      // Sort notes by depth (painter's algorithm)
      const sorted = [...this.notes].sort((a, b) => b.z - a.z);

      for (let i = 0; i < sorted.length; i++) {
        const n = sorted[i];
        const tex = this.billCanvases[n.type];
        if (!tex) continue;

        this.ctx.save();
        this.ctx.translate(n.x, n.y);

        // 3D perspective projection transform simulation
        const cosYaw = Math.cos(n.yaw);
        const cosPitch = Math.cos(n.pitch);

        this.ctx.rotate(n.roll);
        // Non-uniform scaling to simulate 3D tumbling in space
        this.ctx.scale(cosYaw, cosPitch);

        // Holographic sheen reflection highlight based on tumbling angle
        const lightAngle = Math.abs(cosYaw * cosPitch);
        this.ctx.globalAlpha = n.opacity * (0.65 + lightAngle * 0.35);

        // Draw the pre-rendered banknote
        const w = n.width;
        const h = n.height;
        this.ctx.drawImage(tex, -w * 0.5, -h * 0.5, w, h);

        // Subtle specular foil shimmer across banknote surface
        if (lightAngle > 0.72) {
          const shimmer = this.ctx.createLinearGradient(-w * 0.5, -h * 0.5, w * 0.5, h * 0.5);
          shimmer.addColorStop(0, 'rgba(255, 255, 255, 0)');
          shimmer.addColorStop(0.5, `rgba(0, 229, 255, ${0.18 * (lightAngle - 0.72) * 3.5})`);
          shimmer.addColorStop(1, 'rgba(255, 255, 255, 0)');
          this.ctx.fillStyle = shimmer;
          this.ctx.fillRect(-w * 0.5, -h * 0.5, w, h);
        }

        this.ctx.restore();
      }
    }

    /**
     * Controls stream density / speed on demand
     */
    setIntensity(level) {
      if (level === 'subtle') {
        this.options.speed = 0.7;
        this.notes.forEach(n => n.opacity = Math.min(n.opacity, 0.45));
      } else if (level === 'blizzard') {
        this.options.speed = 2.4;
      } else {
        this.options.speed = 1.15;
      }
    }
  }

  window.CurrencyStreamEngine = CurrencyStreamEngine;
})();
