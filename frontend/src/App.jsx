import { useEffect, useState } from 'react'
import {
  BadgeCheck,
  Calculator,
  ChartNoAxesCombined,
  CircleHelp,
  Gauge,
  Route,
  Truck,
} from 'lucide-react'
import './App.css'
import { predictTariff } from './predict.js'

const numberFormat = new Intl.NumberFormat('en-IN', {
  maximumFractionDigits: 0,
})

const fieldDetails = {
  year: { label: 'Year', unit: 'Calendar year', step: '1' },
  wttons: { label: 'Load weight', unit: 'Metric tonnes', step: '0.1' },
  labour: { label: 'Labour cost', unit: 'Cost units', step: '1' },
  toll: { label: 'Toll charges', unit: 'Cost units', step: '1' },
  distance: { label: 'Distance', unit: 'Kilometres', step: '0.1' },
  fuelcost: { label: 'Fuel cost', unit: 'Cost units', step: '1' },
  miscellaneous: { label: 'Other expenses', unit: 'Cost units', step: '0.1' },
}

const categoryDetails = {
  road: { label: 'Road type', descriptions: { nh: 'National highway', sh: 'State highway' } },
  seasonalimpact: { label: 'Seasonal impact', descriptions: { yes: 'Yes', no: 'No' } },
  demand: { label: 'Demand', descriptions: { high: 'High', normal: 'Normal', low: 'Low', less: 'Less' } },
}

const defaults = {
  year: 2015,
  road: 'nh',
  seasonalimpact: 'yes',
  demand: 'high',
  wttons: 10,
  labour: 750,
  toll: 7200,
  distance: 1500,
  fuelcost: 25000,
  miscellaneous: 9000,
}

function Metric({ icon, label, value }) {
  return (
    <div className="metric">
      <div className="metric-icon">{icon}</div>
      <div className="metric-copy">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  )
}

function App() {
  const [artifact, setArtifact] = useState(null)
  const [modelError, setModelError] = useState('')
  const [input, setInput] = useState(defaults)
  const [prediction, setPrediction] = useState(null)
  const [inputError, setInputError] = useState('')

  useEffect(() => {
    let active = true
    fetch(`${import.meta.env.BASE_URL}model.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`Model file returned ${response.status}.`)
        return response.json()
      })
      .then((data) => {
        if (active) setArtifact(data)
      })
      .catch((error) => {
        if (active) setModelError(error.message)
      })
    return () => {
      active = false
    }
  }, [])

  const updateInput = (name, value) => {
    setInput((current) => ({ ...current, [name]: value }))
    setInputError('')
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    if (!artifact) return
    try {
      setPrediction(predictTariff(input, artifact))
      setInputError('')
    } catch (error) {
      setInputError(error.message)
    }
  }

  const fillExample = () => {
    setInput({ ...defaults })
    setPrediction(null)
    setInputError('')
  }

  const cv = artifact?.metrics.crossValidation
  const test = artifact?.metrics.heldOutTest

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Tariffly home">
          <span className="brand-mark"><Route size={19} strokeWidth={2.4} /></span>
          <span>tariffly<span className="brand-dot">.</span></span>
        </a>
        <div className="topbar-right">
          <span className="topbar-caption">FREIGHT INTELLIGENCE</span>
          <span className={`model-status ${modelError ? 'is-error' : ''}`}>
            <span /> {artifact ? 'MODEL READY' : modelError ? 'MODEL UNAVAILABLE' : 'LOADING MODEL'}
          </span>
        </div>
      </header>

      <main className="main-content">
        <section className="intro">
          <div className="intro-copy">
            <div className="eyebrow"><span className="eyebrow-line" /> SMARTER FREIGHT, CLEARER COSTS</div>
            <h1>Know your tariff<br /><span>before you roll.</span></h1>
            <p>Enter your trip details. Get an instant, data-driven estimate for your next truckload.</p>
          </div>
          <div className="intro-art" aria-hidden="true">
            <div className="art-orbit orbit-one" />
            <div className="art-orbit orbit-two" />
            <div className="art-sun" />
            <div className="art-road"><span /><span /><span /></div>
            <div className="art-truck"><Truck size={69} strokeWidth={1.25} /></div>
            <div className="art-leaf leaf-one" />
            <div className="art-leaf leaf-two" />
          </div>
        </section>

        {modelError && (
          <div className="error-banner" role="alert">
            Could not load the browser model: {modelError}
          </div>
        )}

        <div className="workspace">
          <section className="form-panel">
            <div className="section-heading">
              <div>
                <div className="section-kicker">01 — TRIP DETAILS</div>
                <h2>Build your estimate</h2>
                <p>Use your expected trip values to get a tailored forecast.</p>
              </div>
              <button className="text-button" type="button" onClick={fillExample}>Load example</button>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="field-section">
                <div className="field-section-title"><span className="mini-icon"><Truck size={15} /></span>ROUTE & LOAD</div>
                <div className="field-grid">
                  {artifact?.features.categorical.map((feature) => {
                    const details = categoryDetails[feature.name] ?? {
                      label: feature.name,
                      descriptions: {},
                    }
                    return (
                      <label className="field" key={feature.name}>
                        <span className="field-label">{details.label}</span>
                        <span className="select-wrap">
                          <select
                            value={input[feature.name] ?? feature.imputeValue}
                            onChange={(event) => updateInput(feature.name, event.target.value)}
                          >
                            {feature.categories.map((category) => (
                              <option value={category} key={category}>
                                {details.descriptions[category] ?? category}
                              </option>
                            ))}
                          </select>
                        </span>
                      </label>
                    )
                  })}
                  {artifact?.features.numeric.slice(0, 2).map((feature) => {
                    const details = fieldDetails[feature.name] ?? { label: feature.name, unit: '' }
                    return (
                      <label className="field" key={feature.name}>
                        <span className="field-label">{details.label}</span>
                        <span className="input-wrap">
                          <input
                            type="number"
                            step={details.step ?? 'any'}
                            min="0"
                            value={input[feature.name] ?? ''}
                            onChange={(event) => updateInput(feature.name, event.target.value)}
                            required
                          />
                          <span className="field-unit">{details.unit}</span>
                        </span>
                      </label>
                    )
                  })}
                </div>
              </div>

              <div className="field-section cost-section">
                <div className="field-section-title"><span className="mini-icon"><Gauge size={15} /></span>TRIP COSTS & DISTANCE</div>
                <div className="field-grid">
                  {artifact?.features.numeric.slice(2).map((feature) => {
                    const details = fieldDetails[feature.name] ?? { label: feature.name, unit: '' }
                    return (
                      <label className="field" key={feature.name}>
                        <span className="field-label">{details.label}</span>
                        <span className="input-wrap">
                          <input
                            type="number"
                            step={details.step ?? 'any'}
                            min="0"
                            value={input[feature.name] ?? ''}
                            onChange={(event) => updateInput(feature.name, event.target.value)}
                            required
                          />
                          <span className="field-unit">{details.unit}</span>
                        </span>
                      </label>
                    )
                  })}
                </div>
              </div>

              {inputError && <div className="input-error" role="alert">{inputError}</div>}
              <div className="form-footer">
                <span className="privacy-note"><BadgeCheck size={16} /> Your trip details stay in this browser.</span>
                <button className="predict-button" type="submit" disabled={!artifact}>
                  <Calculator size={17} /> {artifact ? 'Calculate estimate' : 'Loading model…'}
                </button>
              </div>
            </form>
          </section>

          <aside className="results-column">
            <section className={`result-card ${prediction !== null ? 'has-result' : ''}`}>
              <div className="result-card-top">
                <div className="section-kicker">02 — YOUR FORECAST</div>
                <span className="result-badge"><span /> LIVE MODEL</span>
              </div>
              {prediction !== null ? (
                <div className="result-content" aria-live="polite">
                  <div className="result-label">Estimated tariff</div>
                  <div className="result-value">{numberFormat.format(Math.round(prediction))}</div>
                  <div className="result-unit">tariff units <span>·</span> point estimate</div>
                  <div className="result-divider" />
                  <div className="result-footnote"><CircleHelp size={15} /> Estimate based on historical trip patterns.</div>
                </div>
              ) : (
                <div className="empty-result">
                  <div className="empty-icon"><Calculator size={23} /></div>
                  <strong>Your estimate starts here</strong>
                  <p>Fill in the trip details and calculate to see your predicted tariff.</p>
                  <div className="empty-decoration"><span /><span /><span /><span /><span /><span /><span /></div>
                </div>
              )}
              <div className="result-card-stamp"><Truck size={17} /><span>FREIGHT COST ESTIMATOR</span></div>
            </section>

            <section className="model-card">
              <div className="model-card-heading">
                <div className="model-icon"><ChartNoAxesCombined size={17} /></div>
                <div><span className="section-kicker">MODEL SNAPSHOT</span><h3>{artifact?.model.name ?? 'Loading model'}</h3></div>
              </div>
              <div className="model-metrics">
                <Metric icon={<ChartNoAxesCombined size={16} />} label="Cross-validation R²" value={cv ? cv.r2.toFixed(2) : '—'} />
                <Metric icon={<Gauge size={16} />} label="Test RMSE" value={test ? numberFormat.format(Math.round(test.rmse)) : '—'} />
              </div>
              <div className="model-disclaimer">Trained on {artifact ? numberFormat.format(artifact.metrics.trainingRows + artifact.metrics.testRows) : '—'} historical trips. Estimates are directional, not a guaranteed quote.</div>
            </section>
          </aside>
        </div>

        <footer className="page-footer">
          <span>© 2026 TARIFFLY · FREIGHT ESTIMATION</span>
          <span>BUILT FOR THE ROAD AHEAD <span className="footer-star">✳</span></span>
        </footer>
      </main>
    </div>
  )
}

export default App
