// src/engine/audio.ts
// Web Audio API engine — soft piano-ish tone using oscillators + exponential decay.

import { type Chord, chordMidi } from './theory';

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private scheduled: AudioNode[] = [];
  private _stepTimer: ReturnType<typeof setInterval> | null = null;

  private _ensure(): void {
    if (this.ctx) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const AC = window.AudioContext ?? (window as any).webkitAudioContext;
    this.ctx = new AC() as AudioContext;
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.55;
    this.master.connect(this.ctx.destination);
  }

  private midiToHz(m: number): number {
    return 440 * Math.pow(2, (m - 69) / 12);
  }

  private playNote(midi: number, when: number, dur: number, gain = 0.18): void {
    const ctx = this.ctx!;
    const master = this.master!;
    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const osc3 = ctx.createOscillator();
    osc1.type = 'triangle';
    osc2.type = 'sine';
    osc3.type = 'sine';
    const hz = this.midiToHz(midi);
    osc1.frequency.value = hz;
    osc2.frequency.value = hz * 2;
    osc3.frequency.value = hz * 0.5;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(gain, when + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, when + dur);
    osc1.connect(g); osc2.connect(g); osc3.connect(g);
    g.connect(master);
    osc1.start(when); osc2.start(when); osc3.start(when);
    osc1.stop(when + dur + 0.05);
    osc2.stop(when + dur + 0.05);
    osc3.stop(when + dur + 0.05);
    this.scheduled.push(osc1, osc2, osc3, g);
  }

  playChord(c: Chord, when: number, dur: number): void {
    const notes = chordMidi(c);
    notes.forEach((n, i) => {
      this.playNote(n, when, dur, i === 0 ? 0.18 : 0.13);
    });
  }

  stop(): void {
    if (!this.ctx) return;
    try { this.master!.gain.cancelScheduledValues(this.ctx.currentTime); } catch { /* noop */ }
    try { this.master!.gain.setValueAtTime(0, this.ctx.currentTime); } catch { /* noop */ }
    this.scheduled.forEach(n => {
      try { (n as OscillatorNode).stop?.(); } catch { /* noop */ }
    });
    this.scheduled = [];
    if (this._stepTimer !== null) clearInterval(this._stepTimer);
    setTimeout(() => { if (this.master) this.master.gain.value = 0.55; }, 50);
  }

  playSequence(
    chords: Chord[],
    tempo: number,
    onStep: (i: number) => void,
    onStop: () => void,
  ): void {
    this._ensure();
    if (this.ctx!.state === 'suspended') this.ctx!.resume();
    this.stop();
    const beatSec = 60 / tempo;
    const stepSec = beatSec * 2; // each chord = 2 beats
    const start = this.ctx!.currentTime + 0.08;
    chords.forEach((c, i) => {
      this.playChord(c, start + i * stepSec, stepSec * 0.92);
    });
    const startMs = performance.now() + 80;
    let idx = -1;
    this._stepTimer = setInterval(() => {
      const elapsed = (performance.now() - startMs) / 1000;
      const newIdx = Math.floor(elapsed / stepSec);
      if (newIdx !== idx) {
        idx = newIdx;
        if (idx >= 0 && idx < chords.length) onStep(idx);
      }
      if (idx >= chords.length) {
        clearInterval(this._stepTimer!);
        this._stepTimer = null;
        onStop();
      }
    }, 20);
  }
}
