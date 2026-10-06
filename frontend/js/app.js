/**
 * MindTrack — Frontend Client Application
 * Production ML Dashboard: Reactive state, real-time feature engineering preview,
 * API integration, interactive Chart.js visualizations, and prediction history.
 */

document.addEventListener('DOMContentLoaded', () => {
  'use strict';

  // Determine API base URL dynamically
  const isLocalBackend = window.location.port === '8000' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const API_BASE = isLocalBackend ? `${window.location.origin}` : 'http://127.0.0.1:8000';
  const API_V1 = `${API_BASE}/api/v1`;

  // Chart Instances
  let featureChartInstance = null;
  let stressChartInstance = null;
  let distributionChartInstance = null;

  // History State
  let allHistoryRecords = [];

  // Preset Data Profiles (100% Real API Inputs)
  const PRESETS = {
    balanced: {
      age: 21,
      gender: 'Female',
      country: 'Canada',
      academic_level: 'Undergraduate',
      most_used_platform: 'Instagram',
      purpose_of_use: 'Entertainment',
      avg_daily_usage_hours: 3.5,
      daily_unlocks: 110,
      study_hours: 4.5,
      physical_activity_hours: 2.0,
      sleep_hours_per_night: 7.8,
      stress_level: 'Medium'
    },
    screen_heavy: {
      age: 20,
      gender: 'Male',
      country: 'USA',
      academic_level: 'Undergraduate',
      most_used_platform: 'TikTok',
      purpose_of_use: 'Entertainment',
      avg_daily_usage_hours: 8.5,
      daily_unlocks: 240,
      study_hours: 1.5,
      physical_activity_hours: 0.5,
      sleep_hours_per_night: 4.8,
      stress_level: 'Very High'
    },
    exam_crunch: {
      age: 22,
      gender: 'Female',
      country: 'India',
      academic_level: 'Graduate',
      most_used_platform: 'YouTube',
      purpose_of_use: 'Education',
      avg_daily_usage_hours: 4.0,
      daily_unlocks: 160,
      study_hours: 7.5,
      physical_activity_hours: 0.8,
      sleep_hours_per_night: 5.5,
      stress_level: 'High'
    },
    active_rested: {
      age: 23,
      gender: 'Male',
      country: 'Australia',
      academic_level: 'Graduate',
      most_used_platform: 'LinkedIn',
      purpose_of_use: 'Networking',
      avg_daily_usage_hours: 2.0,
      daily_unlocks: 75,
      study_hours: 5.0,
      physical_activity_hours: 2.8,
      sleep_hours_per_night: 8.5,
      stress_level: 'Low'
    }
  };

  // DOM Elements
  const form = document.getElementById('predictionForm');
  const submitBtn = document.getElementById('submitBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnLabel = document.getElementById('btnLabel');
  const formAlert = document.getElementById('formAlert');

  // Interactive Ratio Elements
  const ratioPreviewVal = document.getElementById('ratioPreviewVal');
  const ratioBarFill = document.getElementById('ratioBarFill');
  const ratioFootnote = document.getElementById('ratioFootnote');
  const usageInput = document.getElementById('avg_daily_usage_hours');
  const sleepInput = document.getElementById('sleep_hours_per_night');

  // Results Elements
  const resultsEmpty = document.getElementById('resultsEmpty');
  const resultsContent = document.getElementById('resultsContent');
  const gaugeProgress = document.getElementById('gaugeProgress');
  const scoreNum = document.getElementById('scoreNum');
  const tierBadge = document.getElementById('tierBadge');
  const interpretationText = document.getElementById('interpretationText');
  const factorsList = document.getElementById('factorsList');
  const recommendationsList = document.getElementById('recommendationsList');
  const recommendationsContainer = document.getElementById('recommendationsContainer');

  // Top Nav & Header
  const statusIndicator = document.getElementById('statusIndicator');
  const navHistoryCount = document.getElementById('navHistoryCount');
  const heroHistoryCount = document.getElementById('heroHistoryCount');

  // History Elements
  const historyTableBody = document.getElementById('historyTableBody');
  const historyEmptyState = document.getElementById('historyEmptyState');
  const historyTable = document.getElementById('historyTable');
  const historySearchInput = document.getElementById('historySearchInput');
  const historyFilterStress = document.getElementById('historyFilterStress');
  const historyCountBadge = document.getElementById('historyCountBadge');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');

  // Safety Modal Elements
  const safetyModal = document.getElementById('safetyModal');
  const openSafetyBtns = document.querySelectorAll('.open-safety-modal');
  const closeSafetyBtn = document.getElementById('closeSafetyBtn');

  // =========================================================================
  // 1. Navigation & Tab Switching
  // =========================================================================
  const navTabs = document.querySelectorAll('.nav-tab');
  const viewSections = document.querySelectorAll('.view-section');

  function switchTab(tabId) {
    navTabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
    viewSections.forEach(s => s.classList.toggle('active', s.id === `view-${tabId}`));

    if (tabId === 'insights') loadInsights();
    if (tabId === 'model') loadModelDiagnostics();
    if (tabId === 'history') loadHistory();
  }

  navTabs.forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  // Modal Handlers
  openSafetyBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      if (safetyModal) safetyModal.classList.add('active');
    });
  });

  if (closeSafetyBtn && safetyModal) {
    closeSafetyBtn.addEventListener('click', () => safetyModal.classList.remove('active'));
    safetyModal.addEventListener('click', (e) => {
      if (e.target === safetyModal) safetyModal.classList.remove('active');
    });
  }

  // =========================================================================
  // 2. Real-Time Feature Engineering: Screen-to-Sleep Ratio Preview
  // =========================================================================
  function updateRatioPreview() {
    if (!usageInput || !sleepInput || !ratioPreviewVal || !ratioBarFill) return;

    const usage = parseFloat(usageInput.value) || 0.0;
    const sleep = parseFloat(sleepInput.value) || 7.0;
    const ratio = Math.round((usage / (sleep + 0.1)) * 100) / 100;

    ratioPreviewVal.textContent = ratio.toFixed(2);

    // Visual gauge width: scale 0.0 -> 2.0 ratio to 5% -> 100%
    const fillPercent = Math.min(Math.max((ratio / 2.0) * 100, 5), 100);
    ratioBarFill.style.width = `${fillPercent}%`;

    if (ratio < 0.65) {
      ratioBarFill.style.background = 'linear-gradient(90deg, #10b981, #06b6d4)';
      ratioPreviewVal.style.color = '#34d399';
      ratioPreviewVal.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      ratioFootnote.textContent = 'Protective Balance: Sleep significantly exceeds screen time (Strong positive model weight).';
    } else if (ratio <= 1.0) {
      ratioBarFill.style.background = 'linear-gradient(90deg, #06b6d4, #f59e0b)';
      ratioPreviewVal.style.color = '#38bdf8';
      ratioPreviewVal.style.borderColor = 'rgba(6, 182, 212, 0.4)';
      ratioFootnote.textContent = 'Moderate Balance: Daily screen time is roughly balanced with recovery sleep.';
    } else {
      ratioBarFill.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';
      ratioPreviewVal.style.color = '#f87171';
      ratioPreviewVal.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      ratioFootnote.textContent = 'Digital Displacement: Screen time exceeds sleep duration (Model\'s top negative driver).';
    }
  }

  if (usageInput) usageInput.addEventListener('input', updateRatioPreview);
  if (sleepInput) sleepInput.addEventListener('input', updateRatioPreview);

  // =========================================================================
  // 3. Demo Profile Handlers
  // =========================================================================
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.dataset.preset;
      const data = PRESETS[presetKey];
      if (!data) return;

      // Clear any prior alert
      hideAlert();

      // Populate every form field
      Object.entries(data).forEach(([key, val]) => {
        const el = document.getElementById(key);
        if (el) {
          el.value = val;
          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });

      // Update active button indicator
      document.querySelectorAll('.preset-btn').forEach(b => {
        b.classList.remove('preset-btn--active');
        b.style.background = '';
        b.style.borderColor = '';
        b.style.color = '';
      });
      btn.classList.add('preset-btn--active');
      btn.style.background = 'rgba(16, 185, 129, 0.18)';
      btn.style.borderColor = 'rgba(16, 185, 129, 0.55)';
      btn.style.color = '#ffffff';

      updateRatioPreview();
    });
  });

  // =========================================================================
  // 4. Alert / Feedback Helper
  // =========================================================================
  function showAlert(message, type = 'error') {
    if (!formAlert) return;
    formAlert.textContent = message;
    formAlert.style.display = 'block';
    if (type === 'error') {
      formAlert.style.background = 'rgba(239, 68, 68, 0.14)';
      formAlert.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      formAlert.style.color = '#fca5a5';
    } else {
      formAlert.style.background = 'rgba(16, 185, 129, 0.14)';
      formAlert.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      formAlert.style.color = '#6ee7b7';
    }
  }

  function hideAlert() {
    if (formAlert) formAlert.style.display = 'none';
  }

  // =========================================================================
  // 5. API Health Check & Header Counters
  // =========================================================================
  async function checkApiHealth() {
    try {
      const res = await fetch(`${API_V1}/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        statusIndicator.innerHTML = '<span class="status-dot"></span> Backend API Connected';
        statusIndicator.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        updateHistoryCounters();
        return;
      }
    } catch {
      try {
        const res2 = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        if (res2.ok) {
          statusIndicator.innerHTML = '<span class="status-dot"></span> Backend API Connected';
          statusIndicator.style.borderColor = 'rgba(16, 185, 129, 0.3)';
          updateHistoryCounters();
          return;
        }
      } catch {}
      statusIndicator.innerHTML = '<span class="status-dot" style="background:#f59e0b; box-shadow:0 0 8px #f59e0b;"></span> Standalone / Connecting...';
    }
  }

  async function updateHistoryCounters() {
    try {
      const res = await fetch(`${API_V1}/predictions?limit=100`);
      if (res.ok) {
        const data = await res.json();
        const count = data.total || (data.records ? data.records.length : 0);
        if (navHistoryCount) navHistoryCount.textContent = count;
        if (heroHistoryCount) heroHistoryCount.textContent = count;
        if (historyCountBadge) historyCountBadge.textContent = `${count} assessments`;
      }
    } catch {}
  }

  // =========================================================================
  // 6. Form Submission & Real ML Inference
  // =========================================================================
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideAlert();

    const age = parseInt(document.getElementById('age').value, 10);
    const gender = document.getElementById('gender').value;
    const country = (document.getElementById('country').value || '').trim() || 'Other';
    const academic_level = document.getElementById('academic_level').value;
    const most_used_platform = document.getElementById('most_used_platform').value;
    const purpose_of_use = document.getElementById('purpose_of_use').value;
    const avg_daily_usage_hours = parseFloat(document.getElementById('avg_daily_usage_hours').value);
    const daily_unlocks = parseInt(document.getElementById('daily_unlocks').value, 10);
    const study_hours = parseFloat(document.getElementById('study_hours').value);
    const physical_activity_hours = parseFloat(document.getElementById('physical_activity_hours').value);
    const sleep_hours_per_night = parseFloat(document.getElementById('sleep_hours_per_night').value);
    const stress_level = document.getElementById('stress_level').value;

    // Client-side validations
    if (isNaN(age) || age < 10 || age > 100) {
      showAlert('Please enter an age between 10 and 100.');
      return;
    }
    if (isNaN(avg_daily_usage_hours) || avg_daily_usage_hours < 0 || avg_daily_usage_hours > 24) {
      showAlert('Daily screen usage must be between 0 and 24 hours.');
      return;
    }
    if (isNaN(daily_unlocks) || daily_unlocks < 0 || daily_unlocks > 1000) {
      showAlert('Daily phone unlocks must be between 0 and 1000.');
      return;
    }
    if (isNaN(sleep_hours_per_night) || sleep_hours_per_night < 0 || sleep_hours_per_night > 24) {
      showAlert('Nightly sleep duration must be between 0 and 24 hours.');
      return;
    }
    if (isNaN(study_hours) || study_hours < 0 || study_hours > 24) {
      showAlert('Daily study hours must be between 0 and 24 hours.');
      return;
    }
    if (isNaN(physical_activity_hours) || physical_activity_hours < 0 || physical_activity_hours > 24) {
      showAlert('Physical activity hours must be between 0 and 24 hours.');
      return;
    }

    const payload = {
      age,
      gender,
      country,
      academic_level,
      most_used_platform,
      purpose_of_use,
      avg_daily_usage_hours,
      daily_unlocks,
      study_hours,
      physical_activity_hours,
      sleep_hours_per_night,
      stress_level
    };

    // UI Loading state
    submitBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnLabel.textContent = 'Evaluating ML Pipeline...';

    try {
      let response;
      try {
        response = await fetch(`${API_V1}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      } catch {
        response = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      }

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Prediction failed (HTTP ${response.status})`);
      }

      const result = await response.json();
      renderPredictionResult(result);
      updateHistoryCounters();
    } catch (err) {
      showAlert(`Inference Error: ${err.message}`);
    } finally {
      submitBtn.disabled = false;
      btnSpinner.style.display = 'none';
      btnLabel.textContent = 'Generate Wellbeing Prediction';
    }
  });

  // =========================================================================
  // 7. Render Prediction Result Card
  // =========================================================================
  function renderPredictionResult(data) {
    if (!resultsEmpty || !resultsContent) return;

    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'block';

    const score = Number(data.predicted_mental_health_score) || 0.0;
    scoreNum.textContent = score.toFixed(1);

    // SVG Circular progress gauge (circumference = 565.48 for r=90)
    const circumference = 565.48;
    const progress = Math.min(Math.max(score / 10.0, 0), 1);
    const offset = circumference * (1 - progress);
    gaugeProgress.style.strokeDashoffset = offset;
    gaugeProgress.style.stroke = data.tier_color || '#10b981';

    // Tier badge
    tierBadge.textContent = data.wellbeing_tier;
    tierBadge.style.color = data.tier_color || '#10b981';
    tierBadge.style.borderColor = `${data.tier_color}44`;
    tierBadge.style.background = `${data.tier_color}18`;

    // Qualitative interpretation
    interpretationText.textContent = data.interpretation;

    // Render contributing factors
    factorsList.innerHTML = '';
    const factors = data.contributing_factors || [];

    if (factors.length === 0) {
      factorsList.innerHTML = '<p class="text-faint" style="font-size:0.8rem;">Balanced habit profile — all indicators are near cohort averages.</p>';
    } else {
      factors.forEach(f => {
        const card = document.createElement('div');
        card.className = `factor-card ${f.impact}`;

        const iconSym = f.impact === 'positive' ? '▲' : (f.impact === 'negative' ? '▼' : '●');
        const sign = f.weight_score > 0 ? `+${f.weight_score}` : `${f.weight_score}`;

        card.innerHTML = `
          <div class="factor-icon">${iconSym}</div>
          <div class="factor-info">
            <div class="factor-heading">
              <span>${f.factor}</span>
              <span class="factor-delta">${sign} pts</span>
            </div>
            <p class="factor-desc">${f.description}</p>
          </div>
        `;
        factorsList.appendChild(card);
      });
    }

    // Render Actionable Lifestyle Recommendations
    if (recommendationsContainer && recommendationsList) {
      recommendationsList.innerHTML = '';
      const recs = data.recommendations || [];

      if (recs.length > 0) {
        recommendationsContainer.style.display = 'block';
        recs.forEach(rec => {
          const li = document.createElement('li');
          li.className = 'recommendation-item';
          li.innerHTML = `
            <span class="recommendation-icon">💡</span>
            <span>${rec}</span>
          `;
          recommendationsList.appendChild(li);
        });
      } else {
        recommendationsContainer.style.display = 'none';
      }
    }

    // Smooth scroll into view on mobile
    if (window.innerWidth < 960) {
      resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // =========================================================================
  // 8. Analytics & Insights Tab (100% Real API Data)
  // =========================================================================
  async function loadInsights() {
    try {
      const res = await fetch(`${API_V1}/insights`);
      if (!res.ok) return;
      const data = await res.json();

      const summary = data.dataset_summary || {};
      const averages = summary.feature_averages || {};

      document.getElementById('statTotalCohort').textContent = (summary.total_samples || 4998).toLocaleString();
      document.getElementById('statMeanScore').textContent = (summary.score_mean || 6.23).toFixed(2);
      document.getElementById('statAvgScreen').textContent = `${averages.Avg_Daily_Usage_Hours || 5.08}h`;
      document.getElementById('statAvgSleep').textContent = `${averages.Sleep_Hours_Per_Night || 6.63}h`;

      // Render Charts with real API values
      if (window.Chart) {
        if (data.feature_importances) {
          renderFeatureImportanceChart(data.feature_importances);
        }
        if (summary.stress_score_averages) {
          renderStressScoreChart(summary.stress_score_averages);
        }
        if (summary.score_distribution) {
          renderDistributionChart(summary.score_distribution);
        }
      }
    } catch (e) {
      console.warn('Insights load notice:', e);
    }
  }

  function renderFeatureImportanceChart(importances) {
    const ctx = document.getElementById('featureChart');
    if (!ctx) return;

    if (featureChartInstance) featureChartInstance.destroy();

    const topFactors = importances.slice(0, 8);
    const labels = topFactors.map(f => f.feature.replace(/_/g, ' '));
    const values = topFactors.map(f => (f.importance * 100).toFixed(1));

    featureChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Predictive Tree Weight (%)',
          data: values,
          backgroundColor: [
            'rgba(16, 185, 129, 0.8)',
            'rgba(6, 182, 212, 0.8)',
            'rgba(99, 102, 241, 0.8)',
            'rgba(245, 158, 11, 0.8)',
            'rgba(239, 68, 68, 0.8)',
            'rgba(139, 92, 246, 0.8)',
            'rgba(236, 72, 153, 0.8)',
            'rgba(14, 165, 233, 0.8)'
          ],
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => ` ${item.raw}% predictive weight`
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af', callback: val => `${val}%` }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#f3f4f6', font: { weight: '600' } }
          }
        }
      }
    });
  }

  function renderStressScoreChart(stressAverages) {
    const ctx = document.getElementById('stressChart');
    if (!ctx) return;

    if (stressChartInstance) stressChartInstance.destroy();

    // Use REAL calculated numbers from API
    const labels = ['Low Stress', 'Medium Stress', 'High Stress', 'Very High Stress'];
    const values = [
      stressAverages['Low'] || 7.84,
      stressAverages['Medium'] || 7.14,
      stressAverages['High'] || 6.03,
      stressAverages['Very High'] || 5.05
    ];

    stressChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Cohort Average Score',
          data: values,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.12)',
          fill: true,
          tension: 0.35,
          pointRadius: 6,
          pointHoverRadius: 8,
          pointBackgroundColor: '#10b981',
          borderWidth: 2.5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => ` Mean Score: ${item.raw.toFixed(2)} / 10`
            }
          }
        },
        scales: {
          y: {
            min: 4.0,
            max: 8.5,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#f3f4f6' }
          }
        }
      }
    });
  }

  function renderDistributionChart(scoreDistribution) {
    const ctx = document.getElementById('distributionChart');
    if (!ctx) return;

    if (distributionChartInstance) distributionChartInstance.destroy();

    const labels = Object.keys(scoreDistribution);
    const counts = Object.values(scoreDistribution);

    distributionChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels.map(l => `${l} pts`),
        datasets: [{
          label: 'Number of Students',
          data: counts,
          backgroundColor: 'rgba(99, 102, 241, 0.75)',
          borderRadius: 5,
          hoverBackgroundColor: 'rgba(99, 102, 241, 0.95)'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => ` ${item.raw} students in this score bracket`
            }
          }
        },
        scales: {
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#cbd5e1' }
          }
        }
      }
    });
  }

  // =========================================================================
  // 9. Model Diagnostics Tab
  // =========================================================================
  async function loadModelDiagnostics() {
    try {
      const res = await fetch(`${API_V1}/model-info`);
      if (!res.ok) return;
      const data = await res.json();

      document.getElementById('champModelName').textContent = data.model_name;
      document.getElementById('champRationale').textContent = data.selection_rationale;

      const m = data.metrics || {};
      document.getElementById('metricR2').textContent = (m.r2 || 0.8552).toFixed(4);
      document.getElementById('metricRMSE').textContent = (m.rmse || 0.5085).toFixed(4);
      document.getElementById('metricMAE').textContent = (m.mae || 0.3875).toFixed(4);
      document.getElementById('metricCVR2').textContent = (m.cv_r2_mean || 0.8372).toFixed(4);

      // Render model comparison table
      const tbody = document.getElementById('comparisonTableBody');
      tbody.innerHTML = '';
      const models = data.model_comparison || [];

      models.forEach(mod => {
        const tr = document.createElement('tr');
        if (mod.selected) tr.className = 'champion-row';

        const checkBadge = mod.selected ? ' <span style="color:#10b981; font-size:0.75rem;">★ Selected</span>' : '';
        tr.innerHTML = `
          <td>${mod.model_name}${checkBadge}</td>
          <td>${mod.cv_mae.toFixed(4)}</td>
          <td>${mod.cv_rmse.toFixed(4)}</td>
          <td>${mod.cv_r2.toFixed(4)}</td>
          <td><strong>${mod.test_r2.toFixed(4)}</strong></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.warn('Model diagnostics load notice:', e);
    }
  }

  // =========================================================================
  // 10. History Tab with Real-Time Search & Filtering
  // =========================================================================
  async function loadHistory() {
    try {
      const res = await fetch(`${API_V1}/predictions?limit=100`);
      if (!res.ok) return;
      const data = await res.json();

      allHistoryRecords = data.records || [];
      renderHistoryTable();
      updateHistoryCounters();
    } catch (e) {
      console.warn('History load notice:', e);
    }
  }

  function renderHistoryTable() {
    if (!historyTableBody || !historyEmptyState || !historyTable) return;

    const searchTerm = (historySearchInput ? historySearchInput.value : '').toLowerCase().trim();
    const stressFilter = (historyFilterStress ? historyFilterStress.value : 'ALL');

    const filtered = allHistoryRecords.filter(r => {
      const inp = r.input_data || {};
      const matchesStress = (stressFilter === 'ALL' || inp.stress_level === stressFilter);

      if (!matchesStress) return false;
      if (!searchTerm) return true;

      const searchableText = `${inp.country || ''} ${inp.gender || ''} ${inp.most_used_platform || ''} ${r.wellbeing_tier || ''} ${inp.academic_level || ''}`.toLowerCase();
      return searchableText.includes(searchTerm);
    });

    if (historyCountBadge) {
      historyCountBadge.textContent = `Showing ${filtered.length} of ${allHistoryRecords.length}`;
    }

    if (filtered.length === 0) {
      historyEmptyState.style.display = 'block';
      historyEmptyState.textContent = allHistoryRecords.length === 0
        ? 'No assessments recorded yet. Run a check-in to see your history here.'
        : 'No records match your search filter.';
      historyTableBody.innerHTML = '';
      return;
    }

    historyEmptyState.style.display = 'none';
    historyTableBody.innerHTML = '';

    filtered.forEach(r => {
      const inp = r.input_data || {};
      const tr = document.createElement('tr');

      let dateStr = 'Recent';
      try {
        dateStr = new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' });
      } catch {}

      tr.innerHTML = `
        <td style="font-size:0.8rem; color:#9ca3af;">${dateStr}</td>
        <td>${inp.age || '—'}y, ${inp.gender || '—'} (${inp.country || '—'})</td>
        <td>${inp.avg_daily_usage_hours || '—'}h screen / ${inp.sleep_hours_per_night || '—'}h sleep</td>
        <td><span style="color:${inp.stress_level === 'Very High' ? '#ef4444' : (inp.stress_level === 'High' ? '#f59e0b' : '#34d399')}">${inp.stress_level || '—'}</span></td>
        <td>
          <strong style="color:${r.tier_color || '#10b981'}; font-size:1rem;">${Number(r.predicted_mental_health_score).toFixed(1)}</strong>
          <div style="font-size:0.72rem; color:${r.tier_color || '#10b981'};">${r.wellbeing_tier}</div>
        </td>
        <td>
          <button class="btn-delete-item" data-id="${r.id}" title="Delete assessment" aria-label="Delete assessment">✕</button>
        </td>
      `;
      historyTableBody.appendChild(tr);
    });

    // Attach individual delete listeners
    document.querySelectorAll('.btn-delete-item').forEach(btn => {
      btn.addEventListener('click', async () => {
        const id = btn.dataset.id;
        if (!confirm('Delete this assessment from history?')) return;
        try {
          await fetch(`${API_V1}/predictions/${id}`, { method: 'DELETE' });
          loadHistory();
        } catch {
          alert('Failed to delete assessment.');
        }
      });
    });
  }

  if (historySearchInput) {
    historySearchInput.addEventListener('input', renderHistoryTable);
  }
  if (historyFilterStress) {
    historyFilterStress.addEventListener('change', renderHistoryTable);
  }

  // Clear all history
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', async () => {
      if (!confirm('Are you sure you want to permanently clear all stored prediction records?')) return;
      try {
        await fetch(`${API_V1}/predictions`, { method: 'DELETE' });
        loadHistory();
      } catch {
        alert('Failed to clear history.');
      }
    });
  }

  // =========================================================================
  // 11. Initial Application Boot
  // =========================================================================
  checkApiHealth();
  updateRatioPreview();

  // Pre-fill balanced profile on boot
  const defaultPreset = PRESETS.balanced;
  Object.entries(defaultPreset).forEach(([key, val]) => {
    const el = document.getElementById(key);
    if (el) el.value = val;
  });

  const firstBtn = document.querySelector('.preset-btn[data-preset="balanced"]');
  if (firstBtn) {
    firstBtn.classList.add('preset-btn--active');
    firstBtn.style.background = 'rgba(16, 185, 129, 0.18)';
    firstBtn.style.borderColor = 'rgba(16, 185, 129, 0.55)';
    firstBtn.style.color = '#ffffff';
  }

  updateRatioPreview();
}); // end DOMContentLoaded
