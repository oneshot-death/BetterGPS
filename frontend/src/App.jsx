
import { useMemo, useState } from 'react'
import {
  LayoutDashboard, Upload, Map, ChartNoAxesCombined,
  History, Settings, Menu, Bell, Search, Route,
  MapPin, Clock, Activity, FileUp, CheckCircle2,
  Navigation, X, AlertTriangle, ShieldAlert
} from 'lucide-react'

import {
  MapContainer, TileLayer, Polyline, CircleMarker, Popup
} from 'react-leaflet'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, CartesianGrid
} from 'recharts'

import 'leaflet/dist/leaflet.css'
import './App.css'

// Default preview points before a GPS file is uploaded
const sampleGps = [
  [12.9345, 77.5345],
  [12.9350, 77.5360],
  [12.9358, 77.5370],
  [12.9365, 77.5385],
  [12.9375, 77.5390],
  [12.9380, 77.5410],
  [12.9390, 77.5420],
  [12.9400, 77.5435]
]

const navItems = [
  { label: 'Dashboard', icon: LayoutDashboard },
  { label: 'Upload GPS', icon: Upload },
  { label: 'Route Map', icon: Map },
  { label: 'Analysis', icon: ChartNoAxesCombined },
  { label: 'Trip History', icon: History }
]

// Calculate distance between two GPS coordinates using Haversine formula
function calculateDistance(point1, point2) {
  const earthRadius = 6371
  const toRadians = degrees => degrees * Math.PI / 180

  const lat1 = toRadians(point1[0])
  const lat2 = toRadians(point2[0])
  const deltaLat = toRadians(point2[0] - point1[0])
  const deltaLon = toRadians(point2[1] - point1[1])

  const a =
    Math.sin(deltaLat / 2) ** 2 +
    Math.cos(lat1) *
    Math.cos(lat2) *
    Math.sin(deltaLon / 2) ** 2

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))

  return earthRadius * c
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '--'

  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainingSeconds = Math.floor(seconds % 60)

  if (hours > 0) return `${hours}h ${minutes}m`
  if (minutes > 0) return `${minutes}m ${remainingSeconds}s`
  return `${remainingSeconds}s`
}

// Detect gaps between consecutive GPS timestamps
function calculateGapStats(records) {
  const expectedInterval = 1
  const gapThreshold = 1.5

  let gapCount = 0
  let missingSamples = 0
  let longestGap = 0
  let outOfOrderCount = 0

  const gaps = []

  for (let i = 1; i < records.length; i++) {
    const previousTime = records[i - 1].timestamp
    const currentTime = records[i].timestamp
    const interval = currentTime - previousTime

    if (interval <= 0) {
      outOfOrderCount++
      continue
    }

    if (interval > gapThreshold) {
      gapCount++

      const estimatedMissing = Math.max(
        0,
        Math.round(interval / expectedInterval) - 1
      )

      missingSamples += estimatedMissing
      longestGap = Math.max(longestGap, interval)

      gaps.push({
        afterRecord: i,
        interval,
        missingSamples: estimatedMissing
      })
    }
  }

  return {
    gapCount,
    missingSamples,
    longestGap,
    outOfOrderCount,
    gaps
  }
}

// Calculate actual trip statistics
function calculateTripStats(records, points) {
  let totalDistance = 0

  for (let i = 1; i < points.length; i++) {
    totalDistance += calculateDistance(points[i - 1], points[i])
  }

  const timestamps = records.map(record => record.timestamp)
  const startTime = Math.min(...timestamps)
  const endTime = Math.max(...timestamps)
  const duration = endTime - startTime

  const averageSpeed = duration > 0
    ? totalDistance / (duration / 3600)
    : 0

  return {
    distance: totalDistance,
    duration,
    averageSpeed,
    start: points[0],
    end: points[points.length - 1],
    pointCount: points.length,
    startTime,
    endTime
  }
}

// Calculate speed for consecutive GPS records. Intervals over 1.5 seconds
// are excluded so a missing-data gap is not treated as normal movement.
function calculateSpeedStats(records) {
  const expectedMaxInterval = 1.5
  const speeds = []
  let skippedGapSegments = 0

  for (let i = 1; i < records.length; i++) {
    const previous = records[i - 1]
    const current = records[i]
    const elapsed = current.timestamp - previous.timestamp

    if (elapsed <= 0) continue
    if (elapsed > expectedMaxInterval) {
      skippedGapSegments++
      continue
    }

    const distance = calculateDistance(
      [previous.latitude, previous.longitude],
      [current.latitude, current.longitude]
    )

    const speed = (distance * 3600) / elapsed

    if (Number.isFinite(speed)) {
      speeds.push({
        time: Number((current.timestamp - records[0].timestamp).toFixed(1)),
        speed: Number(speed.toFixed(2))
      })
    }
  }

  if (speeds.length === 0) {
    return {
      averageSpeed: null,
      minimumSpeed: null,
      maximumSpeed: null,
      segmentCount: 0,
      skippedGapSegments,
      chartData: []
    }
  }

  const values = speeds.map(item => item.speed)
  const total = values.reduce((sum, value) => sum + value, 0)

  return {
    averageSpeed: total / values.length,
    minimumSpeed: Math.min(...values),
    maximumSpeed: Math.max(...values),
    segmentCount: speeds.length,
    skippedGapSegments,
    chartData: speeds
  }
}


// Flag suspicious segment speeds using a robust median/MAD threshold,
// plus a conservative absolute threshold of 200 km/h. These are flags,
// not automatic deletions or proof that a GPS record is incorrect.
function calculateOutlierStats(records) {
  const segments = []
  const maxPlausibleSpeed = 200
  const maxInterval = 1.5

  for (let i = 1; i < records.length; i++) {
    const previous = records[i - 1]
    const current = records[i]
    const elapsed = current.timestamp - previous.timestamp

    if (elapsed <= 0 || elapsed > maxInterval) continue

    const distance = calculateDistance(
      [previous.latitude, previous.longitude],
      [current.latitude, current.longitude]
    )
    const speed = (distance * 3600) / elapsed

    if (Number.isFinite(speed)) {
      segments.push({
        recordIndex: i + 1,
        timestamp: current.timestamp,
        elapsed,
        speed
      })
    }
  }

  if (segments.length === 0) {
    return { count: 0, threshold: maxPlausibleSpeed, outliers: [], analysedSegments: 0 }
  }

  const values = segments.map(segment => segment.speed).sort((a, b) => a - b)
  const median = values.length % 2 === 0
    ? (values[values.length / 2 - 1] + values[values.length / 2]) / 2
    : values[Math.floor(values.length / 2)]

  const deviations = values.map(value => Math.abs(value - median)).sort((a, b) => a - b)
  const mad = deviations.length % 2 === 0
    ? (deviations[deviations.length / 2 - 1] + deviations[deviations.length / 2]) / 2
    : deviations[Math.floor(deviations.length / 2)]

  const robustThreshold = mad > 0
    ? median + (6 * 1.4826 * mad)
    : Math.max(median * 3, maxPlausibleSpeed)
  const threshold = Math.min(Math.max(robustThreshold, maxPlausibleSpeed), 500)

  const outliers = segments
    .filter(segment => segment.speed > threshold || segment.speed > maxPlausibleSpeed)
    .map(segment => ({
      ...segment,
      reason: segment.speed > maxPlausibleSpeed
        ? 'Above 200 km/h reference threshold'
        : 'Unusually high compared with trip speed distribution'
    }))
    .sort((a, b) => b.speed - a.speed)

  return {
    count: outliers.length,
    threshold,
    outliers,
    analysedSegments: segments.length
  }
}


// Read whitespace- or comma-separated numeric files used by the backend.
function parseNumericFile(content, expectedColumns, fileLabel) {
  const lines = String(content)
    .trim()
    .split(/\r?\n/)
    .filter(line => line.trim())

  if (lines.length === 0) {
    throw new Error(`${fileLabel} file is empty.`)
  }

  return lines.map((line, index) => {
    const values = line.trim().split(/[,\s]+/).map(Number)

    if (
      values.length < expectedColumns ||
      values.slice(0, expectedColumns).some(value => !Number.isFinite(value))
    ) {
      throw new Error(`Invalid data in ${fileLabel} file at line ${index + 1}.`)
    }

    return values.slice(0, expectedColumns)
  })
}

// Convert backend arc IDs into separate Leaflet [latitude, longitude] paths.
// A null arc marks a break between disconnected route segments.
function buildPredictedRoute(predictedArcs, nodes, arcs) {
  const routeSegments = []
  let currentSegment = []

  const addPoint = point => {
    const previous = currentSegment[currentSegment.length - 1]
    if (!previous || previous[0] !== point[0] || previous[1] !== point[1]) {
      currentSegment.push(point)
    }
  }

  const finishSegment = () => {
    if (currentSegment.length > 1) routeSegments.push(currentSegment)
    currentSegment = []
  }

  for (const rawArcId of predictedArcs) {
    if (rawArcId === null) {
      finishSegment()
      continue
    }

    const arcId = Number(rawArcId)
    if (!Number.isInteger(arcId) || arcId < 0 || arcId >= arcs.length) {
      throw new Error(`Backend returned an invalid road arc ID: ${rawArcId}`)
    }

    const [fromNode, toNode] = arcs[arcId]
    if (
      !Number.isInteger(fromNode) ||
      !Number.isInteger(toNode) ||
      !nodes[fromNode] ||
      !nodes[toNode]
    ) {
      throw new Error(`Road arc ${arcId} refers to a missing node.`)
    }

    const [fromLongitude, fromLatitude] = nodes[fromNode]
    const [toLongitude, toLatitude] = nodes[toNode]

    // Leaflet coordinates are [latitude, longitude].
    addPoint([fromLatitude, fromLongitude])
    addPoint([toLatitude, toLongitude])
  }

  finishSegment()
  return routeSegments
}

function App() {
  const [active, setActive] = useState('Dashboard')
  const [mobileMenu, setMobileMenu] = useState(false)
  const [file, setFile] = useState(null)
  const [tripPoints, setTripPoints] = useState(sampleGps)
  const [tripStats, setTripStats] = useState(null)
  const [gapStats, setGapStats] = useState(null)
  const [speedStats, setSpeedStats] = useState(null)
  const [outlierStats, setOutlierStats] = useState(null)
  const [originalRecords, setOriginalRecords] = useState([])
  const [showCleaned, setShowCleaned] = useState(false)
  const [message, setMessage] = useState('')

  // Backend map-matching integration
  const [nodesFile, setNodesFile] = useState(null)
  const [arcsFile, setArcsFile] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionError, setPredictionError] = useState('')

  const center = useMemo(() => {
    if (!tripPoints.length) return [12.9345, 77.5345]
    return tripPoints[Math.floor(tripPoints.length / 2)]
  }, [tripPoints])

  // Upload and parse the actual .track file
  function handleFileUpload(event) {
    const selected = event.target.files?.[0]
    if (!selected) return

    if (!selected.name.toLowerCase().endsWith('.track')) {
      setMessage('Please upload a valid .track file.')
      event.target.value = ''
      return
    }

    const reader = new FileReader()

    reader.onload = () => {
      try {
        const lines = String(reader.result)
          .trim()
          .split(/\r?\n/)
          .filter(line => line.trim())

        const records = lines.map(line => {
          const values = line.trim().split(/[,\s]+/).map(Number)

          return {
            longitude: values[0],
            latitude: values[1],
            timestamp: values[2]
          }
        }).filter(record =>
          Number.isFinite(record.longitude) &&
          Number.isFinite(record.latitude) &&
          Number.isFinite(record.timestamp) &&
          Math.abs(record.latitude) <= 90 &&
          Math.abs(record.longitude) <= 180
        )

        if (records.length < 2) {
          setMessage(
            'Could not read enough valid GPS records. Check the file format.'
          )
          return
        }

        const points = records.map(record => [
          record.latitude,
          record.longitude
        ])

        const stats = calculateTripStats(records, points)
        const detectedGaps = calculateGapStats(records)
        const detectedSpeeds = calculateSpeedStats(records)
        const detectedOutliers = calculateOutlierStats(records)

        setFile(selected)
        setPrediction(null)
        setPredictionError('')
        setTripPoints(points)
        setTripStats(stats)
        setGapStats(detectedGaps)
        setSpeedStats(detectedSpeeds)
        setOutlierStats(detectedOutliers)
        setOriginalRecords(records)
        setShowCleaned(false)
        setActive('Route Map')

        setMessage(
          `Successfully loaded ${points.length.toLocaleString()} GPS points from ${selected.name}.`
        )
      } catch (error) {
        console.error(error)
        setMessage('Unable to read this GPS file. Please check its format.')
      }
    }

    reader.onerror = () => {
      setMessage('Error reading the selected file.')
    }

    reader.readAsText(selected)
    event.target.value = ''
  }


  function handleNetworkFileUpload(event, expectedExtension, setSelectedFile) {
    const selected = event.target.files?.[0]
    if (!selected) return

    if (!selected.name.toLowerCase().endsWith(expectedExtension)) {
      setPredictionError(`Please select a valid ${expectedExtension} file.`)
      event.target.value = ''
      return
    }

    setSelectedFile(selected)
    setPrediction(null)
    setPredictionError('')
    event.target.value = ''
  }

  async function handleRunPrediction() {
    if (!file || !nodesFile || !arcsFile) {
      setPredictionError('Please upload the .track, .nodes and .arcs files first.')
      return
    }

    setPredictionLoading(true)
    setPredictionError('')
    setPrediction(null)

    try {
      const [trackText, nodesText, arcsText] = await Promise.all([
        file.text(),
        nodesFile.text(),
        arcsFile.text(),
      ])

      const trackRows = parseNumericFile(trackText, 3, '.track')
      const nodeRows = parseNumericFile(nodesText, 2, '.nodes')
      const arcRows = parseNumericFile(arcsText, 2, '.arcs')

      const track = trackRows.map(([longitude, latitude, timestamp]) => {
        if (Math.abs(longitude) > 180 || Math.abs(latitude) > 90) {
          throw new Error('The .track file contains coordinates outside valid ranges.')
        }
        return { longitude, latitude, timestamp }
      })

      const nodes = nodeRows.map(([longitude, latitude]) => {
        if (Math.abs(longitude) > 180 || Math.abs(latitude) > 90) {
          throw new Error('The .nodes file contains coordinates outside valid ranges.')
        }
        return { longitude, latitude }
      })

      const arcs = arcRows.map(([from_node, to_node]) => {
        if (
          !Number.isInteger(from_node) ||
          !Number.isInteger(to_node) ||
          from_node < 0 ||
          to_node < 0 ||
          from_node >= nodes.length ||
          to_node >= nodes.length
        ) {
          throw new Error('The .arcs file contains a node index that does not exist.')
        }
        return { from_node, to_node }
      })

      const tripId = file.name.replace(/\.track$/i, '')

      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trip_id: tripId, track, nodes, arcs }),
      })

      const responseData = await response.json().catch(() => ({}))

      if (!response.ok) {
        throw new Error(
          responseData.detail ||
          responseData.message ||
          `Prediction request failed with HTTP ${response.status}.`
        )
      }

      // Accept either a directly returned prediction or a nested result object.
      const result = responseData.result || responseData.prediction || responseData

      if (!Array.isArray(result.predicted_arcs)) {
        throw new Error('The backend response did not contain a predicted_arcs array.')
      }

      const routeSegments = buildPredictedRoute(result.predicted_arcs, nodeRows, arcRows)

      if (routeSegments.length === 0) {
        throw new Error('The backend returned no drawable route segments.')
      }

      setPrediction({
        ...result,
        routeSegments,
      })
    } catch (error) {
      console.error('Map-matching request failed:', error)
      setPredictionError(error.message || 'Unable to run map matching.')
    } finally {
      setPredictionLoading(false)
    }
  }

  const statDistance = tripStats
    ? `${tripStats.distance.toFixed(2)} km`
    : '--'

  const statDuration = tripStats
    ? formatDuration(tripStats.duration)
    : '--'

  const statSpeed = tripStats
    ? `${tripStats.averageSpeed.toFixed(2)} km/h`
    : '--'

  // Create a reversible preview by excluding only the endpoint records
  // of segments flagged by the outlier detector. Original records remain intact.
  const flaggedRecordIndexes = new Set(
    (outlierStats?.outliers || []).map(item => item.recordIndex - 1)
  )
  const cleanedPoints = originalRecords
    .filter((record, index) => !flaggedRecordIndexes.has(index))
    .map(record => [record.latitude, record.longitude])

  let cleanedDistance = 0
  for (let i = 1; i < cleanedPoints.length; i++) {
    cleanedDistance += calculateDistance(cleanedPoints[i - 1], cleanedPoints[i])
  }
  const removedPointCount = originalRecords.length - cleanedPoints.length
  const originalDistance = tripStats?.distance ?? 0
  const distanceDifference = originalDistance - cleanedDistance

  return (
    <div className="app-shell">

      {/* Sidebar */}
      <aside className={`sidebar ${mobileMenu ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-icon"><Route size={23} /></div>
          <div>
            <h2>PathSnap</h2>
            <span>SMART ROUTE INTELLIGENCE</span>
          </div>
          <button
            className="mobile-close"
            onClick={() => setMobileMenu(false)}
          >
            <X />
          </button>
        </div>

        <div className="nav-caption">WORKSPACE</div>

        <nav>
          {navItems.map(item => {
            const Icon = item.icon

            return (
              <button
                key={item.label}
                className={`nav-item ${active === item.label ? 'active' : ''}`}
                onClick={() => {
                  setActive(item.label)
                  setMobileMenu(false)
                }}
              >
                <Icon size={19} />
                <span>{item.label}</span>
              </button>
            )
          })}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className="status-dot" />
            Frontend prototype
            <span className="demo-label">DEMO</span>
          </div>

          <button
            className="nav-item settings-item"
            onClick={() => setActive('Settings')}
          >
            <Settings size={19} /> Settings
          </button>

          <div className="profile">
            <div className="avatar">PS</div>
            <div>
              <strong>PathSnap User</strong>
              <span>Project workspace</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="topbar">
          <button
            className="menu-toggle"
            onClick={() => setMobileMenu(true)}
          >
            <Menu />
          </button>

          <div className="breadcrumb">
            Workspace <span>/</span> {active}
          </div>

          <div className="top-actions">
            <div className="demo-pill"><span /> GPS ANALYSIS</div>
            <button className="icon-button" aria-label="Search">
              <Search size={19} />
            </button>
            <button className="icon-button" aria-label="Notifications">
              <Bell size={19} />
            </button>
            <div className="top-avatar">PS</div>
          </div>
        </header>

        <div className="page-content">

          {/* Page Heading */}
          <section className="welcome-row">
            <div>
              <div className="eyebrow">GPS INTELLIGENCE PLATFORM</div>
              <h1>
                {active === 'Dashboard' ? 'Route Analytics Dashboard' : active}
              </h1>
              <p>
                Explore GPS trajectories, analyse road networks and visualise routes.
              </p>
            </div>

            <label className="primary-button upload-button">
              <FileUp size={18} /> Upload GPS file
              <input
                type="file"
                accept=".track"
                onChange={handleFileUpload}
                hidden
              />
            </label>
          </section>

          {/* Upload Message */}
          {message && (
            <div className="message-banner">
              <CheckCircle2 size={18} />
              {message}
              <button onClick={() => setMessage('')} aria-label="Dismiss">
                <X size={17} />
              </button>
            </div>
          )}

          {/* Statistics */}
          <section className="stats-grid">
            <StatCard
              icon={MapPin}
              label="GPS Points"
              value={tripStats ? tripStats.pointCount.toLocaleString() : '--'}
              note={file ? 'Actual uploaded GPS records' : 'Upload a GPS file'}
            />

            <StatCard
              icon={Route}
              label="GPS Distance"
              value={statDistance}
              note="Calculated from GPS coordinates"
            />

            <StatCard
              icon={Clock}
              label="Trip Duration"
              value={statDuration}
              note="Calculated from timestamps"
            />

            <StatCard
              icon={Activity}
              label="Average Speed"
              value={statSpeed}
              note="GPS distance / trip duration"
            />
          </section>

          {/* Map and Upload Section */}
          <section className="content-grid">

            <div className="panel map-panel">
              <div className="panel-heading">
                <div>
                  <h3>GPS Trajectory Visualisation</h3>
                  <p>Actual uploaded GPS coordinates</p>
                </div>
                <span className="live-tag"><span /> MAP PREVIEW</span>
              </div>

              <div className="map-legend">
                <span><i className="legend-blue" /> GPS trajectory</span>
                {prediction && (
                  <span>
                    <i style={{ display: 'inline-block', width: 18, height: 4, background: '#8b5cf6', borderRadius: 4, marginRight: 6 }} />
                    Predicted road route
                  </span>
                )}
                <span><i className="legend-green" /> Start point</span>
                {outlierStats && outlierStats.count > 0 && (
                  <span><i style={{ display: 'inline-block', width: 9, height: 9, borderRadius: '50%', background: '#e45b5b', marginRight: 6 }} /> Suspicious point</span>
                )}
              </div>

              <div className="map-frame">
                <MapContainer
                  key={center.join(',')}
                  center={center}
                  zoom={15}
                  scrollWheelZoom={true}
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />

                  <Polyline
                    positions={tripPoints}
                    pathOptions={{
                      color: '#2878e8',
                      weight: 5,
                      opacity: 0.9
                    }}
                  />

                  {prediction?.routeSegments.map((segment, index) => (
                    <Polyline
                      key={`predicted-route-${index}`}
                      positions={segment}
                      pathOptions={{
                        color: '#8b5cf6',
                        weight: 5,
                        opacity: 0.95
                      }}
                    />
                  ))}

                  {showCleaned && cleanedPoints.length > 1 && (
                    <Polyline
                      positions={cleanedPoints}
                      pathOptions={{
                        color: '#16a36a',
                        weight: 4,
                        opacity: 0.95,
                        dashArray: '8 7'
                      }}
                    />
                  )}

                  {tripPoints.length > 0 && (
                    <>
                      <CircleMarker
                        center={tripPoints[0]}
                        radius={7}
                        pathOptions={{
                          color: '#fff',
                          fillColor: '#16a36a',
                          fillOpacity: 1,
                          weight: 3
                        }}
                      >
                        <Popup>Starting point</Popup>
                      </CircleMarker>

                      <CircleMarker
                        center={tripPoints[tripPoints.length - 1]}
                        radius={7}
                        pathOptions={{
                          color: '#fff',
                          fillColor: '#e45b5b',
                          fillOpacity: 1,
                          weight: 3
                        }}
                      >
                        <Popup>Ending point</Popup>
                      </CircleMarker>

                      {/* Highlight the GPS coordinate at the end of each flagged segment */}
                      {outlierStats?.outliers.map(item => {
                        const pointIndex = item.recordIndex - 1
                        const suspiciousPoint = tripPoints[pointIndex]
                        if (!suspiciousPoint) return null

                        return (
                          <CircleMarker
                            key={`outlier-${item.recordIndex}`}
                            center={suspiciousPoint}
                            radius={8}
                            pathOptions={{
                              color: '#ffffff',
                              fillColor: '#e45b5b',
                              fillOpacity: 1,
                              weight: 2
                            }}
                          >
                            <Popup>
                              <strong>Suspicious GPS segment</strong><br />
                              Record: {item.recordIndex}<br />
                              Timestamp: {item.timestamp.toFixed(1)} s<br />
                              Calculated speed: {item.speed.toFixed(2)} km/h<br />
                              {item.reason}
                            </Popup>
                          </CircleMarker>
                        )
                      })}
                    </>
                  )}
                </MapContainer>
              </div>

              <div className="map-footer">
                <span><Navigation size={15} /> Interactive map</span>
                <span>{tripPoints.length.toLocaleString()} GPS coordinates</span>
              </div>
            </div>

            <div className="right-column">

              {/* Upload Card and backend map-matching controls */}
              <div className="panel upload-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Map Matching Input</h3>
                    <p>Upload the three files required by the backend</p>
                  </div>
                  <Upload size={20} />
                </div>

                <label className="drop-zone">
                  <div className="upload-circle"><FileUp size={24} /></div>
                  <strong>{file ? file.name : 'Choose a .track file'}</strong>
                  <span>GPS coordinates and timestamps</span>
                  <input
                    type="file"
                    accept=".track"
                    onChange={handleFileUpload}
                    hidden
                  />
                </label>

                <label className="drop-zone" style={{ marginTop: 10 }}>
                  <div className="upload-circle"><FileUp size={24} /></div>
                  <strong>{nodesFile ? nodesFile.name : 'Choose a .nodes file'}</strong>
                  <span>{nodesFile ? 'Road nodes selected' : 'Road network node coordinates'}</span>
                  <input
                    type="file"
                    accept=".nodes"
                    onChange={event => handleNetworkFileUpload(event, '.nodes', setNodesFile)}
                    hidden
                  />
                </label>

                <label className="drop-zone" style={{ marginTop: 10 }}>
                  <div className="upload-circle"><FileUp size={24} /></div>
                  <strong>{arcsFile ? arcsFile.name : 'Choose an .arcs file'}</strong>
                  <span>{arcsFile ? 'Road arcs selected' : 'Connections between road nodes'}</span>
                  <input
                    type="file"
                    accept=".arcs"
                    onChange={event => handleNetworkFileUpload(event, '.arcs', setArcsFile)}
                    hidden
                  />
                </label>

                <button
                  type="button"
                  className="primary-button"
                  style={{ width: '100%', justifyContent: 'center', marginTop: 14 }}
                  onClick={handleRunPrediction}
                  disabled={!file || !nodesFile || !arcsFile || predictionLoading}
                >
                  {predictionLoading ? 'Running map matching...' : 'Run Map Matching'}
                </button>

                <div className="file-note">
                  .track: longitude, latitude, timestamp · .nodes: longitude, latitude · .arcs: from_node, to_node
                </div>

                {predictionError && (
                  <p role="alert" style={{ color: '#c24141', fontSize: 13, marginTop: 10 }}>
                    {predictionError}
                  </p>
                )}

                {prediction && (
                  <div className="trip-details" style={{ marginTop: 12 }}>
                    <p><strong>Prediction:</strong> Complete</p>
                    <p><strong>Method:</strong> {prediction.prediction_method || 'XGBoost + HMM/Viterbi'}</p>
                    <p><strong>Predicted road arcs:</strong> {prediction.predicted_arcs.filter(arc => arc !== null).length.toLocaleString()}</p>
                    <p><strong>Route segments:</strong> {prediction.routeSegments.length}</p>
                  </div>
                )}
              </div>

              {/* Trip Information */}
              <div className="panel activity-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Trip Information</h3>
                    <p>GPS trajectory details</p>
                  </div>
                </div>

                {tripStats ? (
                  <div className="trip-details">
                    <p><strong>Start:</strong> {tripStats.start[0].toFixed(6)}, {tripStats.start[1].toFixed(6)}</p>
                    <p><strong>End:</strong> {tripStats.end[0].toFixed(6)}, {tripStats.end[1].toFixed(6)}</p>
                    <p><strong>Duration:</strong> {formatDuration(tripStats.duration)}</p>
                    <p><strong>Average speed:</strong> {statSpeed}</p>
                  </div>
                ) : (
                  <div className="empty-state">
                    <MapPin size={25} />
                    <p>Upload a GPS file to view trip information.</p>
                  </div>
                )}
              </div>

              {/* NEW: GPS Gap Analysis */}
              <div className="panel activity-panel">
                <div className="panel-heading">
                  <div>
                    <h3>GPS Gap Analysis</h3>
                    <p>Missing GPS sample detection</p>
                  </div>
                  <AlertTriangle size={20} color="#e59b32" />
                </div>

                {gapStats ? (
                  <div className="trip-details">
                    <p>
                      <strong>Detected gaps:</strong>{' '}
                      {gapStats.gapCount}
                    </p>
                    <p>
                      <strong>Estimated missing samples:</strong>{' '}
                      {gapStats.missingSamples}
                    </p>
                    <p>
                      <strong>Longest gap:</strong>{' '}
                      {formatDuration(gapStats.longestGap)}
                    </p>
                    <p>
                      <strong>Out-of-order timestamps:</strong>{' '}
                      {gapStats.outOfOrderCount}
                    </p>

                    <p style={{ fontSize: '12px', color: '#7b8798' }}>
                      A gap is counted when consecutive timestamps differ
                      by more than 1.5 seconds.
                    </p>
                  </div>
                ) : (
                  <div className="empty-state">
                    <Clock size={25} />
                    <p>Upload a GPS file to analyse timestamp gaps.</p>
                  </div>
                )}
              </div>

              {/* NEW: GPS Speed Analysis */}
              <div className="panel activity-panel speed-analysis-panel">
                <div className="panel-heading">
                  <div>
                    <h3>GPS Speed Analysis</h3>
                    <p>Speed calculated between consecutive GPS points</p>
                  </div>
                  <Activity size={20} color="#2878e8" />
                </div>

                {speedStats ? (
                  <>
                    <div className="trip-details">
                      <p>
                        <strong>Average segment speed:</strong>{' '}
                        {speedStats.averageSpeed === null
                          ? '--'
                          : `${speedStats.averageSpeed.toFixed(2)} km/h`}
                      </p>
                      <p>
                        <strong>Minimum segment speed:</strong>{' '}
                        {speedStats.minimumSpeed === null
                          ? '--'
                          : `${speedStats.minimumSpeed.toFixed(2)} km/h`}
                      </p>
                      <p>
                        <strong>Maximum segment speed:</strong>{' '}
                        {speedStats.maximumSpeed === null
                          ? '--'
                          : `${speedStats.maximumSpeed.toFixed(2)} km/h`}
                      </p>
                      <p>
                        <strong>Segments analysed:</strong> {speedStats.segmentCount}
                      </p>
                      <p>
                        <strong>Segments skipped due to GPS gaps:</strong>{' '}
                        {speedStats.skippedGapSegments}
                      </p>
                    </div>

                    {speedStats.chartData.length > 0 && (
                      <div style={{ width: '100%', height: 230, marginTop: 18 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={speedStats.chartData}
                            margin={{ top: 8, right: 12, left: 0, bottom: 8 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="time"
                              tick={{ fontSize: 11 }}
                              label={{ value: 'Elapsed time (seconds)', position: 'insideBottom', offset: -4, fontSize: 11 }}
                            />
                            <YAxis
                              tick={{ fontSize: 11 }}
                              width={48}
                              label={{ value: 'km/h', angle: -90, position: 'insideLeft', fontSize: 11 }}
                            />
                            <Tooltip
                              formatter={(value) => [`${value} km/h`, 'Speed']}
                              labelFormatter={(value) => `Elapsed time: ${value}s`}
                            />
                            <Line
                              type="monotone"
                              dataKey="speed"
                              stroke="#2878e8"
                              strokeWidth={2}
                              dot={false}
                              activeDot={{ r: 4 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    <p style={{ fontSize: '12px', color: '#7b8798', marginTop: 10 }}>
                      Segments longer than 1.5 seconds are excluded because they
                      cross detected GPS gaps. Extreme speeds may still indicate
                      GPS noise or location outliers.
                    </p>
                  </>
                ) : (
                  <div className="empty-state">
                    <Activity size={25} />
                    <p>Upload a GPS file to calculate segment speeds.</p>
                  </div>
                )}
              </div>

              {/* GPS Outlier Detection */}
              <div className="panel activity-panel">
                <div className="panel-heading">
                  <div>
                    <h3>GPS Outlier Detection</h3>
                    <p>Potentially abnormal movement segments</p>
                  </div>
                  <ShieldAlert size={20} color="#e59b32" />
                </div>

                {outlierStats ? (
                  <div className="trip-details">
                    <p>
                      <strong>Flagged segments:</strong> {outlierStats.count}
                    </p>
                    <p>
                      <strong>Segments analysed:</strong> {outlierStats.analysedSegments}
                    </p>
                    <p>
                      <strong>Robust speed threshold:</strong>{' '}
                      {outlierStats.threshold.toFixed(2)} km/h
                    </p>

                    {outlierStats.outliers.length > 0 ? (
                      <div style={{ marginTop: 12 }}>
                        <strong>Highest flagged segments</strong>
                        <div style={{ overflowX: 'auto', marginTop: 8 }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                            <thead>
                              <tr style={{ textAlign: 'left', color: '#7b8798' }}>
                                <th style={{ padding: '7px 5px' }}>Record</th>
                                <th style={{ padding: '7px 5px' }}>Time (s)</th>
                                <th style={{ padding: '7px 5px' }}>Speed</th>
                              </tr>
                            </thead>
                            <tbody>
                              {outlierStats.outliers.slice(0, 8).map(item => (
                                <tr key={item.recordIndex}>
                                  <td style={{ padding: '7px 5px', borderTop: '1px solid #edf0f5' }}>{item.recordIndex}</td>
                                  <td style={{ padding: '7px 5px', borderTop: '1px solid #edf0f5' }}>{item.timestamp.toFixed(1)}</td>
                                  <td style={{ padding: '7px 5px', borderTop: '1px solid #edf0f5', fontWeight: 600 }}>{item.speed.toFixed(2)} km/h</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ) : (
                      <p style={{ color: '#16845b', marginTop: 10 }}>
                        No unusually high-speed segments were flagged.
                      </p>
                    )}

                    <p style={{ fontSize: 12, color: '#7b8798', marginTop: 12 }}>
                      Flags use a median/MAD statistical check and a 200 km/h reference threshold. Review flagged points before excluding them; they are not automatically removed.
                    </p>
                  </div>
                ) : (
                  <div className="empty-state">
                    <ShieldAlert size={25} />
                    <p>Upload a GPS file to check for suspicious segments.</p>
                  </div>
                )}
              </div>

            </div>
          </section>

          {/* Original versus cleaned trajectory preview */}
          <section className="panel bottom-panel" style={{ marginTop: 20 }}>
            <div className="panel-heading">
              <div>
                <h3>GPS Trajectory Cleaning</h3>
                <p>Compare the original trajectory with an outlier-excluded preview</p>
              </div>
              <span className="completed-tag">Preview only</span>
            </div>

            {tripStats && outlierStats ? (
              <>
                <div className="trip-details">
                  <p><strong>Original GPS points:</strong> {originalRecords.length.toLocaleString()}</p>
                  <p><strong>Points excluded in preview:</strong> {removedPointCount.toLocaleString()}</p>
                  <p><strong>Points retained:</strong> {cleanedPoints.length.toLocaleString()}</p>
                  <p><strong>Original GPS distance:</strong> {originalDistance.toFixed(2)} km</p>
                  <p><strong>Preview distance:</strong> {cleanedDistance.toFixed(2)} km</p>
                  <p><strong>Distance difference:</strong> {distanceDifference.toFixed(2)} km</p>
                </div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 14 }}>
                  <button
                    type="button"
                    className="primary-button"
                    onClick={() => setShowCleaned(value => !value)}
                  >
                    {showCleaned ? 'Hide cleaned preview' : 'Show cleaned preview on map'}
                  </button>
                  <span style={{ alignSelf: 'center', fontSize: 12, color: '#7b8798' }}>
                    Blue = original · Green dashed = cleaned preview
                  </span>
                </div>
                <p style={{ fontSize: 12, color: '#7b8798', marginTop: 12 }}>
                  This preview excludes the endpoint record of each flagged segment. It does not modify or overwrite the uploaded file. Removing points can create straight-line connections across excluded locations, so review the map before using this as cleaned data.
                </p>
              </>
            ) : (
              <div className="empty-state">
                <MapPin size={25} />
                <p>Upload a GPS file to compare the original and cleaned trajectory.</p>
              </div>
            )}
          </section>

          {/* Trip Summary */}
          <section className="panel bottom-panel">
            <div className="panel-heading">
              <div>
                <h3>Current Trip</h3>
                <p>Uploaded GPS trajectory overview</p>
              </div>
              <span className="completed-tag">
                {file ? 'GPS data loaded' : 'Awaiting upload'}
              </span>
            </div>

            <div className="trip-row">
              <div className="trip-icon"><Route size={20} /></div>

              <div className="trip-info">
                <strong>{file ? file.name : 'No trip uploaded'}</strong>
                <span>
                  {file ? 'Actual GPS file' : 'Upload a .track file to begin'}
                </span>
              </div>

              <span className="trip-points">
                {tripStats ? `${tripStats.pointCount.toLocaleString()} points` : '--'}
              </span>

              <span className="completed-tag">
                {file ? 'Preview ready' : 'Not started'}
              </span>
            </div>
          </section>

          <footer>
            © 2026 PathSnap · Distributed Map-Matching & Route Reconstruction
          </footer>
        </div>
      </main>
    </div>
  )
}

function StatCard({ icon: Icon, label, value, note }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{label}</span>
        <div className="stat-icon"><Icon size={19} /></div>
      </div>
      <h2>{value}</h2>
      <p>{note}</p>
    </div>
  )
}

export default App
