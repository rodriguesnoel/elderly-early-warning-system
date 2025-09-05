// static/js/app.js

document.getElementById('sensorForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const formData = new FormData(e.target);
  const data = Object.fromEntries(formData);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: new URLSearchParams(formData)
    });

    const result = await response.json();

    if (result.error) {
      alert('Error: ' + result.error);
      return;
    }

    // Update prediction and confidence
    document.getElementById('prediction').textContent = result.prediction;
    document.getElementById('confidence').textContent = (result.confidence * 100).toFixed(1) + '%';

    // --- Render Confidence Gauge ---
    const gaugeDiv = document.getElementById('confidence-gauge');
    const confidencePercent = result.confidence_percent;
    const color = confidencePercent > 70 ? '#4CAF50' : confidencePercent > 40 ? '#FF9800' : '#F44336';

    gaugeDiv.innerHTML = `
      <div style="background: #eee; width: 100%; height: 20px; border-radius: 10px; overflow: hidden;">
        <div style="width: ${confidencePercent}%; height: 100%; background: ${color}; transition: width 0.5s ease;"></div>
      </div>
      <small>${confidencePercent}% confident</small>
    `;

    // --- CHECK FOR ENVIRONMENTAL RISKS ---
    let hasDangerousCondition = false;
    let dangerMessage = '';

    if (data.Temperature === 'Below 12°C - Dangerously Cold') {
      hasDangerousCondition = true;
      dangerMessage += '⚠️ Dangerously cold temperature detected. Risk of hypothermia.<br>';
    }

    if (data.Humidity === 'Below 20% - Dangerously Dry' || data.Humidity === 'Above 80% - Dangerously Humid') {
      hasDangerousCondition = true;
      dangerMessage += '⚠️ Extreme humidity levels detected. Risk of respiratory issues or skin problems.<br>';
    }

    if (data.CO2_ElectroChemicalSensor === '1000–1500 - High' || data.CO2_ElectroChemicalSensor === '1500–2000 - Very High') {
      hasDangerousCondition = true;
      dangerMessage += '⚠️ High CO₂ levels detected. Poor ventilation; risk of drowsiness.<br>';
    }

    if (data.CO2_InfraredSensor === '200–250 - Very High') {
      hasDangerousCondition = true;
      dangerMessage += '⚠️ High CO₂ signal detected. Possible air quality issue.<br>';
    }

    if (data.HVAC_Operation_Mode === 'off' && data.Temperature === 'Below 12°C - Dangerously Cold') {
      hasDangerousCondition = true;
      dangerMessage += '⚠️ HVAC is off during dangerously cold conditions. Heating not active.<br>';
    }

    if (data.Ambient_Light_Level === 'Dim' && data.Temperature === 'Below 12°C - Dangerously Cold') {
      hasDangerousCondition = true;
      dangerMessage += '⚠️ Dim lighting in cold conditions. Increased risk of falls.<br>';
    }

    if ((data.CO2_ElectroChemicalSensor === '1000–1500 - High' || data.CO2_ElectroChemicalSensor === '1500–2000 - Very High') && data.Ambient_Light_Level === 'Dim') {
      hasDangerousCondition = true;
      dangerMessage += '⚠️ High CO₂ in dim lighting — possible poor ventilation during sleep.<br>';
    }

    // --- SET ACTION BASED ON PREDICTION AND ENVIRONMENT ---
    let actionText = '';

    if (hasDangerousCondition) {
      actionText = `
        ⚠️ ENVIRONMENTAL HAZARD DETECTED<br>
        ${dangerMessage}
        👉 IMMEDIATELY check on the resident.<br>
        👉 Ensure heating is turned on and room is warm.<br>
        👉 Provide extra blankets if needed.<br>
        👉 Monitor for signs of shivering, confusion, or discomfort.<br>
        👉 Document the incident and review sensor data.
      `;
    } else if (result.prediction === 'High Activity') {
      actionText = `
        ⚠️ HIGH ACTIVITY DETECTED – POTENTIAL EMERGENCY<br>
        This may indicate a fall, sudden movement, or medical event.<br>
        👉 IMMEDIATELY check on the resident.<br>
        👉 If unresponsive or injured, call emergency services (999).<br>
        👉 Notify family or healthcare provider.<br>
        👉 Document the incident and review sensor data.
      `;
    } else if (result.prediction === 'Moderate Activity') {
      actionText = `
        ✅ Resident is active and likely engaged in normal daily activities.<br>
        No intervention needed.<br>
        Continue regular monitoring.
      `;
    } else {
      actionText = `
        ✅ Normal behavior detected.<br>
        Monitor resident's condition over the next 30 minutes.<br>
        If inactivity continues beyond 60 minutes, check on them to ensure they are well.<br>
        No urgent action required at this time.
      `;
    }

    document.getElementById('action-text').innerHTML = actionText.trim();
    document.getElementById('result').style.display = 'block';

    // --- RENDER CHARTS ---
    renderFeatureImportance(result.feature_importance);
    renderClassDistribution(result.class_distribution);

  } catch (err) {
    console.error('Fetch error:', err);
    alert('Connection error. Is the server running?\n' + err.message);
  }
});

// Render: Top Features Bar Chart (Responsive + Color-Coded)
function renderFeatureImportance(data) {
  const labels = data.map(d => prettifyFeatureName(d.feature));
  const values = data.map(d => d.importance);

  // Generate distinct colors for each bar
  const colors = [
    '#3498db', // Blue
    '#9b59b6', // Purple
    '#e74c3c', // Red
    '#f39c12', // Orange
    '#1abc9c'  // Teal
  ];

  const trace = {
    y: labels.reverse(),
    x: values.reverse(),
    type: 'bar',
    orientation: 'h',
    marker: { color: colors.reverse() },  // Each bar a different color ✅
    text: values.reverse(),
    textposition: 'outside',
    textfont: { size: 10 }
  };

  const layout = {
    title: {
      text: '<b>What Influenced This Prediction?</b>',
      font: { size: 14 }
    },
    margin: { l: 150, r: 20, t: 40, b: 50 },
    xaxis: { 
      title: 'Relative Importance',
      automargin: true
    },
    yaxis: { 
      automargin: true,
      tickfont: { size: 11 }
    },
    showlegend: false,
    autosize: true,
    height: 280
  };

  // Render plot
  Plotly.newPlot('importance-chart', [trace], layout, { 
    displayModeBar: false 
  });

  // Make responsive
  window.onresize = () => {
    Plotly.Plots.resize(document.getElementById('importance-chart'));
    Plotly.Plots.resize(document.getElementById('class-dist-chart'));
  };
}

// Render: Class Distribution Pie Chart (Responsive)
function renderClassDistribution(data) {
  const labels = data.map(d => d.class);
  const values = data.map(d => d.percentage);

  const trace = {
    labels: labels,
    values: values,
    type: 'pie',
    textinfo: 'label+percent',
    insidetextorientation: 'radial',
    marker: {
      colors: ['#4e79a7', '#f28e2b', '#e15759']
    }
  };

  const layout = {
    title: {
      text: '<b>Activity Levels in Resident Population</b>',
      font: { size: 14 }
    },
    margin: { l: 20, r: 20, t: 40, b: 20 },
    showlegend: true,
    autosize: true,
    height: 280
  };

  Plotly.newPlot('class-dist-chart', [trace], layout, { 
    displayModeBar: false 
  });
}

// Optional: Improve feature name readability
function prettifyFeatureName(name) {
  const clean = name
    .replace(/_/g, ' ')
    .replace(/heating active|cooling active|fan only|off/g, '')
    .replace(/dim|normal|bright/g, '')
    .trim();

  const prettyNames = {
    'Temperature': '🌡️ Temperature',
    'Humidity': '💧 Humidity',
    'CO2 ElectroChemicalSensor': 'CO₂ Level',
    'CO2 InfraredSensor': 'CO₂ Signal',
    'HVAC Operation Mode': 'HVAC Mode',
    'Ambient Light Level': 'Lighting',
    'MetalOxideSensor Unit3': 'Heat/Motion Signal',
    'CO GasSensor': 'Carbon Monoxide'
  };

  return prettyNames[clean] || clean;
}