//! Conservative source accounting. The frozen planned reduction is committed
//! after each successful volume microstep; mutable per-cell rates are not reduced again.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct SourceLedger {
    pub pending: f32,
    pub requested: f32,
    pub emitted: f32,
    pub available: f32,
    pub factor: f32,
    pub event_requested: f32,
    pub event_emitted: f32,
    pub fault: u32,
    pub requested_compensation: f32,
    pub emitted_compensation: f32,
    pub event_balance_residual: f32,
    pub event_pending_before: f32,
    pub pending_compensation: f32,
    pub continuous_planned_rate: f32,
}

/// Preserve every f32 operation boundary, including the sign of a zero carry.
#[inline]
pub fn add_compensated(value: f32, total: f32, compensation: f32) -> (f32, f32) {
    let increment = value - compensation;
    let next = total + increment;
    let error = if total.abs() >= increment.abs() {
        (total - next) + increment
    } else {
        (increment - next) + total
    };
    (next, -error)
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SourceCommit {
    pub emitted_measure: f32,
    pub balance_residual: f32,
    pub accepted: bool,
}

impl SourceLedger {
    /// Admit an outer-frame request without changing material or losing a carry.
    pub fn plan(
        &mut self,
        dt: f64,
        requested_rate: f64,
        available: f32,
        factor: f32,
        continuous_planned_rate: f32,
    ) -> Result<(), &'static str> {
        if !dt.is_finite()
            || dt <= 0.0
            || !requested_rate.is_finite()
            || !available.is_finite()
            || !factor.is_finite()
            || !continuous_planned_rate.is_finite()
        {
            return Err("source plan must contain finite rates and a positive timestep");
        }
        let requested = (requested_rate * dt) as f32;
        if !requested.is_finite() {
            return Err("source request overflow");
        }
        let before = self.pending;
        (self.pending, self.pending_compensation) =
            add_compensated(requested, self.pending, self.pending_compensation);
        (self.requested, self.requested_compensation) =
            add_compensated(requested, self.requested, self.requested_compensation);
        self.available = available;
        self.factor = factor;
        self.event_requested = requested;
        self.event_emitted = 0.0;
        self.fault = 0;
        self.event_balance_residual = 0.0;
        self.event_pending_before = before;
        self.continuous_planned_rate = continuous_planned_rate;
        Ok(())
    }

    pub fn commit_microstep(&mut self, dt: f64) -> Result<SourceCommit, &'static str> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err("source commit timestep must be finite and positive");
        }
        // TS/host dt is f64; multiplication rounds once into the f32 ledger lane.
        let emitted_measure = (self.continuous_planned_rate as f64 * dt) as f32;
        if !emitted_measure.is_finite() {
            return Err("source emission overflow");
        }
        let before = self.pending;
        (self.pending, self.pending_compensation) =
            add_compensated(-emitted_measure, before, self.pending_compensation);
        (self.emitted, self.emitted_compensation) =
            add_compensated(emitted_measure, self.emitted, self.emitted_compensation);
        let residual = (self.pending + emitted_measure) - before;
        let accepted = (self.pending as f64)
            >= -8.0 * (f32::EPSILON as f64) * (before.abs().max(emitted_measure.abs()) as f64);
        self.event_emitted += emitted_measure;
        self.event_balance_residual = residual;
        if !accepted {
            self.fault = 2;
        }
        Ok(SourceCommit {
            emitted_measure,
            balance_residual: residual,
            accepted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_increments_survive_outer_frame_carries() {
        let mut total = 0.0;
        let mut carry = 0.0;
        for value in [16_777_216.0, 1.0, 1.0, -16_777_216.0, -0.25, 0.125, 0.125] {
            (total, carry) = add_compensated(value, total, carry);
        }
        assert_eq!(total, 2.0);
        assert_eq!(carry.to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn published_totals_keep_signed_compensation() {
        let (total, carry) = add_compensated(-8.824240684509277, 111.5223617553711, 0.0);
        assert_eq!(total, 102.6981201171875);
        assert_eq!(carry, -9.5367431640625e-7);
        let (next, next_carry) = add_compensated(-8.824240684509277, total, carry);
        assert_eq!(next, 93.8738784790039);
        assert_eq!(next_carry, -0.0000019073486328125);
    }

    #[test]
    fn commits_use_the_frozen_rate_and_record_overdraw() {
        let mut ledger = SourceLedger::default();
        ledger.plan(0.25, 4.0, 1.0, 1.0, 4.0).unwrap();
        assert!(ledger.commit_microstep(0.125).unwrap().accepted);
        assert!(ledger.commit_microstep(0.125).unwrap().accepted);
        assert_eq!(ledger.pending, 0.0);
        assert_eq!(ledger.emitted, 1.0);
        assert!(!ledger.commit_microstep(0.125).unwrap().accepted);
        assert_eq!(ledger.fault, 2);
    }
}
