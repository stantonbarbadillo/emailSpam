import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  BarController,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

// Register the necessary Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  BarController,
  Title,
  Tooltip,
  Legend
);

function EmailClassifier() {
  const [emailContent, setEmailContent] = useState('');
  const [classification, setClassification] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [visualizations, setVisualizations] = useState(null);
  const chartRef = useRef(null);

  const classifyEmail = async () => {
    try {
      const response = await axios.post('http://localhost:5000/classify', {
        email_content: emailContent,
      });
      setClassification(response.data.classification);
    } catch (error) {
      console.error('There was an error classifying the email!', error);
    }
  };

  const trainModel = async () => {
    try {
      await axios.post('http://localhost:5000/train');
      const response = await axios.get('http://localhost:5000/metrics');
      setMetrics(response.data);
    } catch (error) {
      console.error('There was an error training the model!', error);
    }
  };

  useEffect(() => {
    const fetchVisualizations = async () => {
      try {
        const response = await axios.get('http://localhost:5000/visualizations');
        setVisualizations(response.data);
      } catch (error) {
        console.error('There was an error fetching the visualizations!', error);
      }
    };
    fetchVisualizations();
  }, []);

  const barData = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    datasets: [
      {
        label: 'Model Metrics',
        data: metrics ? [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1] : [],
        backgroundColor: ['rgba(75, 192, 192, 0.6)'],
      },
    ],
  };

  useEffect(() => {
    let chartInstance = null;
    if (chartRef.current) {
      const chart = chartRef.current;
      if (chartInstance) {
        chartInstance.destroy();
      }
      chartInstance = new ChartJS(chart, {
        type: 'bar',
        data: barData,
        options: {
          responsive: true,
          plugins: {
            legend: {
              position: 'top',
            },
            title: {
              display: true,
              text: 'Model Metrics',
            },
          },
        },
      });
    }
    return () => {
      if (chartInstance) {
        chartInstance.destroy();
      }
    };
  }, [metrics]);

  return (
    <div>
      <textarea
        value={emailContent}
        onChange={(e) => setEmailContent(e.target.value)}
        placeholder="Paste your email content here"
      />
      <button onClick={classifyEmail}>Classify Email</button>
      {classification && <p>Classification: {classification}</p>}
      <button onClick={trainModel}>Train Model</button>
      {metrics && (
        <div>
          <h3>Model Metrics:</h3>
          <canvas ref={chartRef}></canvas>
        </div>
      )}
      {visualizations && (
        <div>
          <h3>Visualizations</h3>
          <img src={`data:image/png;base64,${visualizations.data_distribution}`} alt="Data Distribution" />
          <img src={`data:image/png;base64,${visualizations.model_performance}`} alt="Model Performance" />
          <img src={`data:image/png;base64,${visualizations.feature_importance}`} alt="Feature Importance" />
        </div>
      )}
    </div>
  );
}

export default EmailClassifier;
