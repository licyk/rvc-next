// Generated from the OpenAPI schema by `python scripts/dev.py typegen`; do not edit.
/** Ranges, defaults and help of the voice and stream parameters, as the server validates them. */
export interface ParamSpec {
  type: string;
  default: unknown;
  min?: number;
  max?: number;
  options?: string[];
  help?: string;
}

export const PARAM_SPECS: Record<'voice' | 'stream', Record<string, ParamSpec>> = {
  "voice": {
    "speaker_id": {
      "type": "integer",
      "default": 0,
      "min": 0,
      "max": 109,
      "help": "Speaker of a multi-speaker voice"
    },
    "pitch": {
      "type": "number",
      "default": 0.0,
      "min": -24,
      "max": 24,
      "help": "Pitch shift in semitones"
    },
    "formant": {
      "type": "number",
      "default": 0.0,
      "min": -2,
      "max": 2,
      "help": "Formant shift in semitones"
    },
    "f0_method": {
      "type": "string",
      "default": "rmvpe",
      "options": [
        "pm",
        "rmvpe",
        "fcpe",
        "crepe",
        "crepe-tiny",
        "swift"
      ],
      "help": "Pitch extraction method"
    },
    "index_rate": {
      "type": "number",
      "default": 0.75,
      "min": 0,
      "max": 1,
      "help": "Index strength: how much of the voice's own timbre is drawn from the index"
    },
    "protect": {
      "type": "number",
      "default": 0.33,
      "min": 0,
      "max": 0.5,
      "help": "Consonant protection; 0.5 turns it off"
    },
    "rms_mix_rate": {
      "type": "number",
      "default": 0.25,
      "min": 0,
      "max": 1,
      "help": "Loudness match; 1 keeps the converted loudness unchanged"
    },
    "unvoiced": {
      "type": "string",
      "default": "protect",
      "options": [
        "protect",
        "zero",
        "original"
      ],
      "help": "Frames without pitch: protect applies to them (protect); they also get no pitch, as classic RVC and Applio (zero); or RVC 2026's rule, where protect has no effect (original)"
    },
    "f0_high_register": {
      "type": "string",
      "default": "off",
      "options": [
        "off",
        "true_pitch",
        "fold"
      ],
      "help": "RMVPE above about 1040 Hz (offline): a second pass corrects its octave errors and writes the true pitch up to the ceiling (true_pitch), or half of it (fold)"
    },
    "f0_ceiling": {
      "type": "number",
      "default": 1250.0,
      "min": 1000,
      "max": 2000,
      "help": "Highest pitch the high-register correction writes, in Hz"
    },
    "autotune": {
      "type": "number",
      "default": 0.0,
      "min": 0,
      "max": 1,
      "help": "Pull notes to the nearest semitone; 0 turns it off, 1 snaps"
    },
    "auto_pitch": {
      "type": "string",
      "default": "off",
      "options": [
        "off",
        "semitone",
        "octave"
      ],
      "help": "Offline: add the key that brings the input's median pitch to the target, in semitones or in whole octaves (which keeps a song's key)"
    },
    "auto_pitch_target": {
      "type": "number",
      "default": 0.0,
      "min": 0,
      "max": 1000,
      "help": "Pitch the automatic key aims at, in Hz; 0: the voice's own (from its training), else 155 Hz"
    }
  },
  "stream": {
    "block_ms": {
      "type": "integer",
      "default": 250,
      "min": 20,
      "max": 1500,
      "help": "Audio processed per step"
    },
    "crossfade_ms": {
      "type": "integer",
      "default": 50,
      "min": 10,
      "max": 150,
      "help": "Crossfade between blocks"
    },
    "context_ms": {
      "type": "integer",
      "default": 2500,
      "min": 50,
      "max": 5000,
      "help": "Past audio the model sees with each block"
    },
    "threshold_db": {
      "type": "number",
      "default": -60.0,
      "min": -60,
      "max": 0,
      "help": "Input gate; -60 turns it off"
    },
    "input_denoise": {
      "type": "boolean",
      "default": false
    },
    "output_denoise": {
      "type": "boolean",
      "default": false
    },
    "denoise_strength": {
      "type": "number",
      "default": 0.9,
      "min": 0,
      "max": 1,
      "help": "How much input and output noise reduction lowers the noise"
    },
    "phase_vocoder": {
      "type": "boolean",
      "default": false,
      "help": "Phase-vocoder crossfade: joins blocks without the dip a plain crossfade can leave"
    }
  }
};
