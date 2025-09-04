// static/js/app.js

document.getElementById('sensorForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);

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

    document.getElementById('prediction').textContent = result.prediction;
    document.getElementById('confidence').textContent = (result.confidence * 100).toFixed(1) + '%';
    document.getElementById('result').style.display = 'block';
  } catch (err) {
    alert('Connection error. Is the server running?');
  }
});