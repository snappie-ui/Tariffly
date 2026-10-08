import { describe, expect, it } from 'vitest'
import artifact from '../public/model.json'
import { predictTariff } from './predict.js'

describe('predictTariff', () => {
  it('matches the Python-exported example prediction', () => {
    expect(predictTariff(artifact.example.input, artifact)).toBeCloseTo(
      artifact.example.prediction,
      8,
    )
  })

  it('uses the fitted imputers when fields are blank', () => {
    const input = Object.fromEntries(
      artifact.features.order.map((name) => [name, '']),
    )
    expect(Number.isFinite(predictTariff(input, artifact))).toBe(true)
  })

  it('rejects invalid numeric input', () => {
    const input = { ...artifact.example.input, year: 'not a year' }
    expect(() => predictTariff(input, artifact)).toThrow(
      'year must be a valid number.',
    )
  })
})
