/**
 * AD-TRACE: Alzheimer's Protein Association Analysis
 * Client Application Logic
 */

// Application State
const AppState = {
  currentAnalysis: null,
  activeTab: 'overview',
  activeView: 'analyze',
  exampleProteins: [],
  networkZoom: 1.0,
  networkPan: { x: 0, y: 0 },
  molViewer: null,
  isSpinning: false,
  charts: {}
};

// Standard 20 Amino Acids
const STANDARD_20_AA = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'];

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSequenceInput();
  initTabs();
  initModals();
  checkBackendHealth();
  loadExampleProteins();
});

// ========================================================
// 1. BACKEND HEALTH & INITIAL DATA
// ========================================================
async function checkBackendHealth() {
  const statusEl = document.getElementById('backend-status-pill');
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('API offline');
    const data = await res.json();
    if (statusEl) {
      statusEl.innerHTML = `<span class="status-dot"></span> SOTA: 93.44% Acc • ROC-AUC 0.947`;
    }
  } catch (err) {
    if (statusEl) {
      statusEl.className = 'model-status-pill warning';
      statusEl.innerHTML = `<span class="status-dot" style="background:#f59e0b;"></span> Backend Connecting...`;
    }
  }
}

async function loadExampleProteins() {
  try {
    const res = await fetch('/api/examples');
    if (!res.ok) throw new Error('Failed to load examples');
    const data = await res.json();
    AppState.exampleProteins = data.examples || [];
    renderExamplesView(AppState.exampleProteins);
  } catch (err) {
    console.warn('Could not preload examples:', err);
  }
}

// ========================================================
// 2. NAVIGATION & VIEWS
// ========================================================
function initNavigation() {
  // Sidebar items
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      navItems.forEach(n => n.classList.remove('active'));
      item.classList.add('active');
      const targetView = item.dataset.view;
      switchView(targetView);
    });
  });

  // Sidebar toggle
  const toggleBtn = document.getElementById('sidebar-toggle-btn');
  const sidebar = document.getElementById('app-sidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
    });
  }

  // Header links
  const homeBtn = document.getElementById('header-home-btn');
  if (homeBtn) {
    homeBtn.addEventListener('click', (e) => {
      e.preventDefault();
      document.querySelector('[data-view="analyze"]')?.click();
    });
  }

  // Documentation TOC smooth scrolling
  document.querySelectorAll('.doc-toc-pill').forEach(pill => {
    pill.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = pill.getAttribute('href')?.substring(1);
      const targetEl = document.getElementById(targetId);
      const container = document.querySelector('.workspace-scroll');
      if (targetEl && container) {
        const topPos = targetEl.offsetTop - 30;
        container.scrollTo({ top: topPos, behavior: 'smooth' });
      }
    });
  });
}

function switchView(viewName) {
  AppState.activeView = viewName;
  const allViews = document.querySelectorAll('.view-section');
  allViews.forEach(v => v.style.display = 'none');

  const target = document.getElementById(`view-${viewName}`);
  if (target) {
    target.style.display = 'block';
  }
}

// ========================================================
// 3. SEQUENCE INPUT & VALIDATION
// ========================================================
function initSequenceInput() {
  const textarea = document.getElementById('sequence-textarea');
  const charCount = document.getElementById('char-count');
  const errorMsg = document.getElementById('input-feedback');
  const clearBtn = document.getElementById('clear-seq-btn');
  const analyzeBtn = document.getElementById('analyze-seq-btn');

  // Input listener for real-time character count and validation
  textarea?.addEventListener('input', () => {
    const val = textarea.value;
    const clean = val.replace(/^>.*$/gm, '').replace(/\s+/g, '');
    if (charCount) charCount.textContent = `${clean.length} aa`;
    validateSequenceInput(val, false);
  });

  // Clear button
  clearBtn?.addEventListener('click', () => {
    if (textarea) textarea.value = '';
    if (charCount) charCount.textContent = '0 aa';
    if (errorMsg) errorMsg.style.display = 'none';
    textarea?.focus();
  });

  // Analyze button
  analyzeBtn?.addEventListener('click', () => {
    handleAnalyzeSequence();
  });

  // Example chip buttons
  document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const gene = chip.dataset.gene;
      loadExampleByGene(gene);
    });
  });
}

function validateSequenceInput(rawText, showError = true) {
  const feedback = document.getElementById('input-feedback');
  if (!feedback) return false;

  const lines = rawText.trim().split('\n');
  let seq = '';
  for (const line of lines) {
    if (!line.trim().startsWith('>')) {
      seq += line.trim().replace(/\s+/g, '');
    }
  }

  seq = seq.toUpperCase();

  if (!seq) {
    if (showError) {
      showFeedback('Please enter or paste a protein amino-acid sequence.', 'error');
    } else {
      feedback.style.display = 'none';
    }
    return false;
  }

  // Check invalid characters
  const validChars = new Set(STANDARD_20_AA);
  const invalid = [];
  for (const c of seq) {
    if (!validChars.has(c)) {
      if (!invalid.includes(c)) invalid.push(c);
    }
  }

  if (invalid.length > 0) {
    showFeedback(`Invalid non-standard characters found: ${invalid.join(', ')}. Supported alphabet: standard 20 amino acids.`, 'error');
    return false;
  }

  feedback.style.display = 'none';
  return true;
}

function showFeedback(msg, type = 'error') {
  const feedback = document.getElementById('input-feedback');
  if (!feedback) return;
  feedback.className = `input-feedback-message ${type}`;
  feedback.textContent = msg;
  feedback.style.display = 'flex';
}

function loadExampleByGene(gene) {
  const example = AppState.exampleProteins.find(e => e.gene.toUpperCase() === gene.toUpperCase());
  const textarea = document.getElementById('sequence-textarea');
  const charCount = document.getElementById('char-count');

  if (example && textarea) {
    textarea.value = `>${example.gene}_HUMAN ${example.protein_name} (UniProt:${example.accession})\n${example.sequence}`;
    if (charCount) charCount.textContent = `${example.sequence_length} aa`;
    validateSequenceInput(textarea.value, false);
    textarea.focus();
  }
}

// ========================================================
// 4. API CALL & RESULTS ORCHESTRATION
// ========================================================
async function handleAnalyzeSequence() {
  const textarea = document.getElementById('sequence-textarea');
  if (!textarea) return;

  const rawText = textarea.value.trim();
  if (!validateSequenceInput(rawText, true)) return;

  showLoading(true);

  try {
    const res = await fetch('/api/analyze-protein', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence: rawText })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error (Status ${res.status})`);
    }

    const data = await res.json();
    AppState.currentAnalysis = data;
    renderAnalysisResults(data);

    // Scroll smoothly to results
    document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });

  } catch (err) {
    showFeedback(`Analysis failed: ${err.message}`, 'error');
  } finally {
    showLoading(false);
  }
}

function showLoading(isLoading) {
  const overlay = document.getElementById('workspace-loading-overlay');
  const analyzeBtn = document.getElementById('analyze-seq-btn');
  if (overlay) {
    if (isLoading) overlay.classList.add('active');
    else overlay.classList.remove('active');
  }
  if (analyzeBtn) {
    analyzeBtn.disabled = isLoading;
    analyzeBtn.innerHTML = isLoading ? 
      `<span class="spinner" style="width:16px;height:16px;border-width:2px;display:inline-block;"></span> Analyzing...` :
      `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg> Analyze Sequence`;
  }
}

// ========================================================
// 5. RENDER RESULTS
// ========================================================
function renderAnalysisResults(data) {
  const resultsSec = document.getElementById('results-section');
  if (resultsSec) resultsSec.style.display = 'block';

  renderPredictionSummary(data.prediction);
  renderIdentityResolution(data.identity);
  renderEvidenceOverview(data);
  renderBrainExpressionTab(data.brain_expression, data.identity);
  renderInteractionNetworkTab(data.interactions, data.identity);
  renderSequenceAnalysisTab(data.sequence_properties, data.identity);
  renderStructureViewerTab(data.structure, data.identity);
  renderDetailedReportTab(data);
}

// 5A. PREDICTION SUMMARY
function renderPredictionSummary(pred) {
  const badgeEl = document.getElementById('prediction-badge');
  const probNumEl = document.getElementById('prob-large-number');
  const barFill = document.getElementById('prob-bar-fill');
  const interpEl = document.getElementById('prediction-interpretation');
  const confEl = document.getElementById('prediction-confidence-val');

  if (!pred) return;

  // Badge classification
  if (badgeEl) {
    if (pred.label.includes('Alzheimer-associated')) {
      badgeEl.className = 'prediction-badge badge-alzheimer';
      badgeEl.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg> ${pred.label}`;
    } else if (pred.label.includes('Control')) {
      badgeEl.className = 'prediction-badge badge-control';
      badgeEl.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> ${pred.label}`;
    } else {
      badgeEl.className = 'prediction-badge badge-inconclusive';
      badgeEl.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> ${pred.label}`;
    }
  }

  // Score
  if (probNumEl) {
    probNumEl.textContent = pred.probability.toFixed(4);
  }

  // Progress Bar
  if (barFill) {
    const pct = Math.min(100, Math.max(0, pred.probability * 100));
    barFill.style.width = `${pct}%`;
    barFill.className = `prob-bar-fill ${pred.probability >= 0.5 ? 'alzheimer' : 'control'}`;
  }

  // Interpretation & Confidence
  if (interpEl) {
    interpEl.textContent = pred.interpretation;
  }
  if (confEl) {
    confEl.textContent = `Distance Confidence: ${pred.confidence_metric.toFixed(4)} (Threshold: ${pred.threshold})`;
  }
}

// 5B. PROTEIN IDENTITY RESOLUTION
function renderIdentityResolution(ident) {
  const matchEl = document.getElementById('ident-best-match');
  const accEl = document.getElementById('ident-accession');
  const geneEl = document.getElementById('ident-gene');
  const nameEl = document.getElementById('ident-protein-name');
  const orgEl = document.getElementById('ident-organism');
  const seqIdentEl = document.getElementById('ident-seq-identity');
  const covEl = document.getElementById('ident-coverage');
  const confEl = document.getElementById('ident-match-confidence');
  const alignBtn = document.getElementById('view-alignment-btn');

  if (!ident || ident.status === 'unmatched') {
    if (matchEl) matchEl.innerHTML = `<span class="pill-unmatched">Unmatched</span>`;
    if (accEl) accEl.textContent = 'None';
    if (geneEl) geneEl.textContent = 'None';
    if (nameEl) nameEl.textContent = 'Novel or Uncataloged Sequence';
    if (orgEl) orgEl.textContent = 'Unknown';
    if (seqIdentEl) seqIdentEl.textContent = 'N/A';
    if (covEl) covEl.textContent = 'N/A';
    if (confEl) confEl.textContent = 'No sufficiently reliable protein identity match was found.';
    if (alignBtn) alignBtn.style.display = 'none';
    return;
  }

  if (matchEl) matchEl.innerHTML = `<strong>${ident.best_match}</strong>`;
  if (accEl) {
    accEl.innerHTML = `<a href="https://www.uniprot.org/uniprotkb/${ident.uniprot_accession}" target="_blank" style="color:var(--primary-blue);text-decoration:underline;">${ident.uniprot_accession}</a>`;
  }
  if (geneEl) geneEl.textContent = ident.gene_name || 'N/A';
  if (nameEl) nameEl.textContent = ident.protein_name || 'N/A';
  if (orgEl) orgEl.textContent = ident.organism || 'Homo sapiens';
  if (seqIdentEl) seqIdentEl.textContent = `${ident.sequence_identity}%`;
  if (covEl) covEl.textContent = `${ident.sequence_coverage}%`;
  
  if (confEl) {
    if (ident.sequence_identity >= 99.0) {
      confEl.innerHTML = `<span class="pill-exact">${ident.match_confidence}</span>`;
    } else {
      confEl.innerHTML = `<span class="pill-homology">${ident.match_confidence}</span>`;
    }
  }

  if (alignBtn) {
    if (ident.alignment) {
      alignBtn.style.display = 'inline-flex';
      alignBtn.onclick = () => openAlignmentModal(ident);
    } else {
      alignBtn.style.display = 'none';
    }
  }
}

// 5C. EVIDENCE OVERVIEW TAB
function renderEvidenceOverview(data) {
  // Mini Brain Expression Preview
  const exprMini = document.getElementById('overview-expr-preview');
  if (exprMini) {
    if (data.brain_expression.status === 'available' && data.brain_expression.measurements.length > 0) {
      const top3 = data.brain_expression.measurements.slice(0, 3);
      exprMini.innerHTML = `
        <div style="font-size:0.8125rem;color:var(--navy-600);margin-bottom:0.5rem;">
          Top Baseline Brain Regions (GTEx v10):
        </div>
        ${top3.map(m => `
          <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem;font-size:0.8125rem;">
            <span>${m.region_name}</span>
            <strong>${m.median_tpm} TPM</strong>
          </div>
        `).join('')}
      `;
    } else {
      exprMini.innerHTML = `<div class="empty-state-box" style="padding:1rem;">Evidence unavailable for this protein.</div>`;
    }
  }

  // Mini STRING Interaction Preview
  const stringMini = document.getElementById('overview-network-preview');
  if (stringMini) {
    if (data.interactions.status === 'available' && data.interactions.edges.length > 0) {
      const topEdges = data.interactions.edges.slice(0, 3);
      stringMini.innerHTML = `
        <div style="font-size:0.8125rem;color:var(--navy-600);margin-bottom:0.5rem;">
          Key Physical & Functional Interactors (STRING v12):
        </div>
        ${topEdges.map(e => `
          <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem;font-size:0.8125rem;">
            <span>${e.source} ↔ ${e.target}</span>
            <span style="color:var(--primary-blue);font-weight:600;">Score: ${e.combined_score}</span>
          </div>
        `).join('')}
      `;
    } else {
      stringMini.innerHTML = `<div class="empty-state-box" style="padding:1rem;">Evidence unavailable for this protein.</div>`;
    }
  }

  // Mini Sequence Evidence Preview
  const seqMini = document.getElementById('overview-seq-preview');
  if (seqMini) {
    const p = data.sequence_properties;
    seqMini.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:0.35rem;font-size:0.8125rem;">
        <div style="display:flex;justify-content:space-between;"><span>Sequence Length:</span><strong>${p.sequence_length} aa</strong></div>
        <div style="display:flex;justify-content:space-between;"><span>Molecular Weight:</span><strong>${(p.molecular_weight/1000).toFixed(1)} kDa</strong></div>
        <div style="display:flex;justify-content:space-between;"><span>Isoelectric Point (pI):</span><strong>${p.isoelectric_point}</strong></div>
        <div style="display:flex;justify-content:space-between;"><span>Hydropathy (GRAVY):</span><strong>${p.gravy}</strong></div>
      </div>
    `;
  }
}

// 5D. BRAIN EXPRESSION TAB
function renderBrainExpressionTab(expr, ident) {
  const container = document.getElementById('brain-expression-content');
  if (!container) return;

  if (expr.status !== 'available' || !expr.measurements || expr.measurements.length === 0) {
    container.innerHTML = `
      <div class="empty-state-box">
        <div class="empty-state-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
        </div>
        <h4>Evidence unavailable for this protein.</h4>
        <p style="font-size:0.875rem;">GTEx v10 transcriptomic baseline records are not available for the resolved identifier (${ident?.gene_name || 'None'}).</p>
      </div>
    `;
    return;
  }

  // Render Full Brain Chart & Data Table
  const regions = expr.measurements;
  const labels = regions.map(r => r.region_name);
  const values = regions.map(r => r.median_tpm);

  container.innerHTML = `
    <div class="callout-disclaimer">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;margin-top:2px;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
      <div>
        <strong>Critical Scientific Context:</strong>
        ${expr.disease_differential_note}
      </div>
    </div>

    <div class="chart-wrapper">
      <canvas id="brain-chart-canvas"></canvas>
    </div>

    <div style="margin-top:1.5rem;">
      <h4 style="font-size:0.9375rem;font-weight:700;margin-bottom:0.75rem;">Regional Expression Levels (GTEx v10)</h4>
      <table class="scientific-table">
        <thead>
          <tr>
            <th>Brain Sub-Region</th>
            <th>Median Abundance</th>
            <th>Units</th>
            <th>Relative Abundance</th>
          </tr>
        </thead>
        <tbody>
          ${regions.map(r => `
            <tr>
              <td><strong>${r.region_name}</strong></td>
              <td class="mono">${r.median_tpm}</td>
              <td>${r.unit}</td>
              <td>
                <div style="width:120px;height:8px;background:var(--navy-200);border-radius:9999px;overflow:hidden;">
                  <div style="width:${Math.min(100, (r.median_tpm / Math.max(...values)) * 100)}%;height:100%;background:var(--primary-blue);"></div>
                </div>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;

  // Draw Chart using Chart.js if loaded, or custom canvas
  setTimeout(() => {
    drawBrainExpressionChart('brain-chart-canvas', labels, values);
  }, 50);
}

function drawBrainExpressionChart(canvasId, labels, data) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  if (window.Chart) {
    if (AppState.charts['brain']) {
      AppState.charts['brain'].destroy();
    }
    const ctx = canvas.getContext('2d');
    AppState.charts['brain'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Median Expression (TPM)',
          data: data,
          backgroundColor: '#3b82f6',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'GTEx v10 RNA Baseline Across 13 Human Brain Regions (TPM)' }
        },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'TPM' } },
          x: { ticks: { autoSkip: false, maxRotation: 45, minRotation: 45 } }
        }
      }
    });
  }
}

// 5E. INTERACTION NETWORK TAB
function renderInteractionNetworkTab(interactions, ident) {
  const container = document.getElementById('interaction-network-content');
  if (!container) return;

  if (interactions.status !== 'available' || !interactions.nodes || interactions.nodes.length === 0) {
    container.innerHTML = `
      <div class="empty-state-box">
        <div class="empty-state-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
        </div>
        <h4>Evidence unavailable for this protein.</h4>
        <p style="font-size:0.875rem;">No documented protein-protein interactions retrieved from STRING v12 for '${ident?.gene_name || 'unmatched'}'.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div class="network-container">
      <div class="network-canvas-wrapper" id="network-canvas-host">
        <div class="network-controls-bar">
          <button class="network-btn" id="net-zoom-in" title="Zoom In">+</button>
          <button class="network-btn" id="net-zoom-out" title="Zoom Out">−</button>
          <button class="network-btn" id="net-reset" title="Reset View">⟲</button>
        </div>
        <svg id="network-svg" width="100%" height="100%"></svg>
      </div>
      <div class="network-details-pane" id="network-details-pane">
        <h4 style="font-size:0.9375rem;font-weight:700;">Interaction Details</h4>
        <p style="font-size:0.8125rem;color:var(--navy-500);">Click any node in the network to inspect its interaction score and biological role.</p>
        <div id="selected-node-info">
          <div style="background:#ffffff;padding:0.75rem;border-radius:var(--radius-sm);border:var(--border-light);font-size:0.8125rem;">
            <strong>Central Node:</strong> ${interactions.query_protein}
            <br><strong>Total Partners:</strong> ${interactions.total_interactors - 1}
            <br><strong>Source:</strong> STRING Database v12
          </div>
        </div>
      </div>
    </div>
  `;

  setTimeout(() => {
    drawSvgNetwork('network-svg', interactions);
  }, 50);
}

function drawSvgNetwork(svgId, data) {
  const svg = document.getElementById(svgId);
  if (!svg) return;

  const width = svg.clientWidth || 600;
  const height = svg.clientHeight || 450;
  const centerX = width / 2;
  const centerY = height / 2;

  // Clear previous
  svg.innerHTML = '';

  const nodes = data.nodes;
  const edges = data.edges;

  // Layout: Query protein at center, partners in a circle
  const queryNode = nodes.find(n => n.is_query) || nodes[0];
  const partnerNodes = nodes.filter(n => n !== queryNode);

  const radius = Math.min(width, height) * 0.35;
  const positions = {};

  positions[queryNode.id] = { x: centerX, y: centerY };

  partnerNodes.forEach((node, idx) => {
    const angle = (idx / partnerNodes.length) * 2 * Math.PI;
    positions[node.id] = {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle)
    };
  });

  // Group for pan and zoom
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  g.setAttribute('id', 'network-zoom-group');
  svg.appendChild(g);

  // Draw Edges
  edges.forEach(edge => {
    const p1 = positions[edge.source];
    const p2 = positions[edge.target];
    if (p1 && p2) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', p1.x);
      line.setAttribute('y1', p1.y);
      line.setAttribute('x2', p2.x);
      line.setAttribute('y2', p2.y);
      line.setAttribute('stroke', '#cbd5e1');
      line.setAttribute('stroke-width', Math.max(1, edge.combined_score * 3));
      line.setAttribute('stroke-opacity', '0.7');
      g.appendChild(line);
    }
  });

  // Draw Nodes
  nodes.forEach(node => {
    const pos = positions[node.id] || { x: centerX, y: centerY };
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', pos.x);
    circle.setAttribute('cy', pos.y);
    circle.setAttribute('r', node.is_query ? '18' : '14');
    circle.setAttribute('fill', node.is_query ? '#2563eb' : '#8b5cf6');
    circle.setAttribute('stroke', '#ffffff');
    circle.setAttribute('stroke-width', '2');
    circle.style.cursor = 'pointer';

    circle.addEventListener('click', () => {
      displayNodeDetails(node, edges, data.query_protein);
    });

    g.appendChild(circle);

    // Node text
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', pos.x);
    text.setAttribute('y', pos.y + (node.is_query ? 28 : 24));
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('font-size', '11px');
    text.setAttribute('font-weight', '600');
    text.setAttribute('fill', '#1e293b');
    text.textContent = node.label;
    g.appendChild(text);
  });

  // Controls
  let zoom = 1.0;
  const zoomGroup = document.getElementById('network-zoom-group');
  document.getElementById('net-zoom-in')?.addEventListener('click', () => {
    zoom *= 1.2;
    zoomGroup?.setAttribute('transform', `scale(${zoom})`);
  });
  document.getElementById('net-zoom-out')?.addEventListener('click', () => {
    zoom /= 1.2;
    zoomGroup?.setAttribute('transform', `scale(${zoom})`);
  });
  document.getElementById('net-reset')?.addEventListener('click', () => {
    zoom = 1.0;
    zoomGroup?.setAttribute('transform', `scale(1)`);
  });
}

function displayNodeDetails(node, edges, queryName) {
  const infoPane = document.getElementById('selected-node-info');
  if (!infoPane) return;

  const relevantEdges = edges.filter(e => e.source === node.id || e.target === node.id);

  infoPane.innerHTML = `
    <div style="background:#ffffff;padding:1rem;border-radius:var(--radius-sm);border:var(--border-light);display:flex;flex-direction:column;gap:0.5rem;">
      <h5 style="font-size:1rem;font-weight:700;color:var(--navy-900);">${node.label}</h5>
      <span style="font-size:0.75rem;color:var(--navy-500);">${node.is_query ? 'Primary Query Protein' : 'Interacting Partner'}</span>
      <hr style="border:none;border-top:var(--border-light);">
      <div style="font-size:0.8125rem;">
        <strong>Connections (${relevantEdges.length}):</strong>
        ${relevantEdges.map(e => `
          <div style="margin-top:0.35rem;padding:0.35rem;background:var(--navy-50);border-radius:4px;">
            <span>${e.source} ↔ ${e.target}</span><br>
            <span style="color:var(--primary-blue);font-weight:600;">Combined Score: ${e.combined_score}</span>
            <br><span style="font-size:0.6875rem;color:var(--navy-500);">${e.interaction_type}</span>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

// 5F. BIOPHYSICAL METRIC INFORMATION & SEQUENCE ANALYSIS TAB
const BIOPHYS_METRIC_INFO = {
  length: {
    title: "Sequence Length",
    unit: "Residues",
    getDisplayValue: (p) => `${p.sequence_length} Residues`,
    getBadgeClass: (p) => "badge-blue",
    definition: "Total number of amino acid residues in the primary polypeptide chain.",
    formulaText: "L = Total amino acid count in primary sequence",
    interpretation: (p) => {
      const len = p.sequence_length;
      if (len <= 50) return `Very short peptide (${len} aa). Compact chains (such as Aβ40/42) lack rigid tertiary folding and often exhibit high intrinsic disorder, predisposing them to rapid nucleation into oligomeric assemblies.`;
      if (len <= 150) return `Small protein / compact domain (${len} aa). Common for modular binding domains, heat shock chaperones, and discrete cleavage fragments.`;
      if (len <= 450) return `Medium-length polypeptide (${len} aa). Consistent with standard cytosolic enzymes, regulatory kinases, and physiological Tau isoforms (352–441 aa).`;
      return `Large multi-domain protein (${len} aa). Exhibits complex tertiary folding architectures, multiple post-translational modification sites, and high sensitivity to proteostatic stress (e.g., full-length APP at 695–770 aa).`;
    },
    referenceNorms: [
      { label: "Short Peptides / Amyloid Aβ", range: "< 100 aa" },
      { label: "Average Human Domain", range: "100 – 350 aa" },
      { label: "Full Precursors (APP / Tau)", range: "350 – 770+ aa" }
    ],
    adRelevance: "In Alzheimer's disease pathology, sequence length is a decisive driver of amyloidogenicity. For example, sequential cleavage of APP produces Aβ1-40 and Aβ1-42. The two extra hydrophobic residues in Aβ42 (Ile41, Ala42) lower the nucleation free energy barrier, accelerating neurotoxic oligomer and fibril formation by orders of magnitude compared to Aβ40."
  },
  mw: {
    title: "Molecular Weight",
    unit: "kDa",
    getDisplayValue: (p) => `${(p.molecular_weight/1000).toFixed(1)} kDa (${Number(p.molecular_weight).toLocaleString()} Da)`,
    getBadgeClass: (p) => "badge-purple",
    definition: "Theoretical total molecular mass of the unmodified polypeptide chain.",
    formulaText: "Mw = Σ(Residue Mass) - (L - 1) × 18.015 Da (condensation water loss)",
    interpretation: (p) => {
      const mwKDa = p.molecular_weight / 1000;
      const bbbText = mwKDa > 0.5 
        ? "Exceeds the passive blood-brain barrier (BBB) diffusion threshold (~400–500 Da), requiring active transport or receptor-mediated transcytosis."
        : "Below 500 Da, within the theoretical boundary for passive blood-brain barrier permeability.";
      return `Calculated mass is ${mwKDa.toFixed(1)} kDa (${Number(p.molecular_weight).toLocaleString()} Da). ${bbbText}`;
    },
    referenceNorms: [
      { label: "Small Neurotoxic Peptides (Aβ)", range: "1 – 5 kDa" },
      { label: "Chaperones & Transporters (APOE)", range: "25 – 40 kDa" },
      { label: "Transmembrane Precursors (APP)", range: "100 – 130 kDa" }
    ],
    adRelevance: "Molecular weight distinguishes intact precursor proteins from neurotoxic fragments and high-molecular-weight assemblies. In Alzheimer's brains, soluble Aβ oligomers ranging from 12 kDa (trimers) to 56 kDa (Aβ*56 dodecamers) represent acutely synaptotoxic assemblies that bind postsynaptic NMDA/EphB2 receptors and induce long-term potentiation (LTP) inhibition."
  },
  pi: {
    title: "Theoretical Isoelectric Point (pI)",
    unit: "pH",
    getDisplayValue: (p) => `pI ${p.isoelectric_point}`,
    getBadgeClass: (p) => "badge-teal",
    definition: "The exact environmental pH at which the net electrical charge of the protein equals zero.",
    formulaText: "Q(pH) = Σ [1 / (1 + 10^(pH - pKa))] - Σ [1 / (1 + 10^(pKa - pH))] = 0",
    interpretation: (p) => {
      const pi = p.isoelectric_point;
      if (pi < 6.2) return `Acidic protein (pI ${pi}). Carries a net negative charge at physiological brain pH (7.2–7.4), providing mutual electrostatic repulsion that favors colloidal solubility in healthy interstitial fluid.`;
      if (pi <= 8.0) return `Neutral / near-physiological pI (pI ${pi}). Because pI is close to physiological brain pH (7.2–7.4), net electrostatic charge approaches zero, reducing repulsive forces and predisposing the protein to self-association and precipitation.`;
      return `Basic protein (pI ${pi}). Carries a net positive charge at physiological brain pH (7.2–7.4), promoting strong electrostatic affinity for polyanionic sulfated glycosaminoglycans (GAGs), membrane phospholipids, and nucleic acids.`;
    },
    referenceNorms: [
      { label: "Acidic (Net Negative at pH 7.4)", range: "pI < 6.2" },
      { label: "Neutral / Aggregation-Prone", range: "pI 6.5 – 7.8" },
      { label: "Basic (Net Positive at pH 7.4)", range: "pI > 8.0" }
    ],
    adRelevance: "Isoelectric point governs colloidal stability and electrostatic aggregation barriers. While physiological CSF pH is 7.2–7.4, neuroinflammatory microenvironments and ischemic hypoxia drive parenchymal acidification down towards pH 6.0–6.5. This shifts ambient pH near the pI of Aβ (pI ≈ 5.3), neutralizing negative charges on Asp and Glu residues and accelerating fibril seeding by over tenfold."
  },
  instability: {
    title: "Instability Index (Guruprasad Method)",
    unit: "Index Score",
    getDisplayValue: (p) => `II = ${p.instability_index} (${p.stability_verdict})`,
    getBadgeClass: (p) => p.instability_index < 40 ? "badge-green" : "badge-amber",
    definition: "Statistical prediction of in vitro protein stability based on the occurrence of 400 dipeptide motifs.",
    formulaText: "II = (10 / L) × Σ DIWV(x_i, x_{i+1})  [Guruprasad DIWV matrix]",
    interpretation: (p) => {
      const ii = p.instability_index;
      if (ii < 40) return `Predicted Stable in vitro (II = ${ii} < 40.0). Dipeptide composition confers structural persistence and resistance against spontaneous denaturation.`;
      return `Predicted Potentially Unstable (II = ${ii} ≥ 40.0). Enriched in destabilizing dipeptides, indicating conformational flexibility, structural metastability, or rapid turnover.`;
    },
    referenceNorms: [
      { label: "Predicted Stable (Half-life > 16 h)", range: "II < 40.0" },
      { label: "Predicted Unstable (Half-life < 5 h)", range: "II ≥ 40.0" },
      { label: "Borderline Metastable", range: "II 38.0 – 42.0" }
    ],
    adRelevance: "Metastability is an essential prerequisite for pathogenic amyloid cascade activation. Intrinsically disordered proteins like Tau and cleavage products of APP exhibit elevated instability indices, reflecting structural plasticity that allows transient sampling of partially unfolded molten-globule intermediates that escape chaperone degradation and seed cross-β fibrillar templates."
  },
  gravy: {
    title: "GRAVY (Grand Average of Hydropathy)",
    unit: "Hydropathy Score",
    getDisplayValue: (p) => `${p.gravy} (${p.gravy < 0 ? 'Hydrophilic' : 'Hydrophobic'})`,
    getBadgeClass: (p) => p.gravy < 0 ? "badge-blue" : "badge-amber",
    definition: "Mean hydropathic character across all residues using the Kyte-Doolittle scale (-4.5 for Arg to +4.5 for Ile).",
    formulaText: "GRAVY = (1 / L) × Σ H(x_i)  [Kyte-Doolittle Hydropathy Index]",
    interpretation: (p) => {
      const g = p.gravy;
      if (g < -0.4) return `Strongly hydrophilic (GRAVY = ${g}). Dominated by polar and charged residues; readily solvated in aqueous cytosol and CSF.`;
      if (g <= 0) return `Moderately hydrophilic (GRAVY = ${g}). Typical globular protein balance between solvent-exposed polar exterior and packed hydrophobic core.`;
      if (g <= 0.5) return `Moderately hydrophobic (GRAVY = ${g}). Enriched in non-polar aliphatic residues; indicates propensity for lipid bilayer interaction or self-aggregation.`;
      return `Strongly hydrophobic (GRAVY = ${g}). Consistent with transmembrane helices or high-affinity hydrophobic aggregation cores.`;
    },
    referenceNorms: [
      { label: "Hydrophilic (Soluble / Globular)", range: "GRAVY < 0.0" },
      { label: "Neutral Hydropathic Balance", range: "GRAVY ≈ 0.0" },
      { label: "Hydrophobic (Membrane / Aggregation)", range: "GRAVY > 0.0" }
    ],
    adRelevance: "Hydrophobic collapse provides the primary thermodynamic driving force for pathological amyloid assembly. Even in overall hydrophilic proteins, localized hydrophobic stretches (such as Aβ17-21 'LVFFA' and C-terminal 'GAIIGLMVGGVVIA') drive water expulsion and cross-β sheet hydrogen bonding, forming toxic membrane-permeabilizing pores in neuronal synapses."
  },
  aromaticity: {
    title: "Aromatic Residue Content",
    unit: "Fraction",
    getDisplayValue: (p) => `${p.aromaticity} (${(p.aromaticity * 100).toFixed(1)}%)`,
    getBadgeClass: (p) => "badge-indigo",
    definition: "Relative molar frequency of aromatic side chains (Phenylalanine, Tryptophan, and Tyrosine) in the sequence.",
    formulaText: "Aromaticity = (Count(Phe) + Count(Trp) + Count(Tyr)) / L",
    interpretation: (p) => {
      const arom = p.aromaticity;
      const pct = (arom * 100).toFixed(1);
      if (arom < 0.05) return `Low aromatic content (${pct}%). Folding stability relies primarily on aliphatic hydrophobic packing and electrostatic salt bridges rather than π-stacking.`;
      if (arom <= 0.10) return `Standard human proteome aromatic content (${pct}%). Falls squarely in the physiological range (7%–9%) of stable globular proteins.`;
      return `Enriched aromatic content (${pct}%). High density of conjugated aromatic rings capable of establishing strong π-π stacking networks and steric zipper motifs.`;
    },
    referenceNorms: [
      { label: "Low Aromatic Content", range: "< 5.0%" },
      { label: "Human Proteome Average", range: "7.0% – 9.0%" },
      { label: "Aromatic-Enriched", range: "> 10.0%" }
    ],
    adRelevance: "Aromatic amino acids act as critical steric catalysts for amyloid fibril nucleation. In Amyloid-beta, Phe19 and Phe20 are essential for establishing the steric zipper spine of the cross-β sheet via planar π-π orbital stacking. Similarly, in Tau neurofibrillary tangles, Tyr310 in the third microtubule-binding repeat (R3) provides the geometric anchor required for paired helical filament (PHF) nucleation."
  }
};

function renderSequenceAnalysisTab(p, ident) {
  const container = document.getElementById('sequence-analysis-content');
  if (!container) return;

  const aaLabels = STANDARD_20_AA;
  const aaCounts = aaLabels.map(aa => p.aa_composition[aa]?.count || 0);

  container.innerHTML = `
    <div class="seq-analysis-container">
      <div class="biophys-metrics-grid">
        <div class="metric-card" data-metric="length" title="Click to view details for Length">
          <div class="metric-card-top">
            <span class="metric-title">Length</span>
            <span class="metric-card-info-icon" title="Click for details">ℹ</span>
          </div>
          <div class="metric-value">${p.sequence_length}</div>
          <div class="metric-sub">Residues</div>
        </div>
        <div class="metric-card" data-metric="mw" title="Click to view details for Molecular Weight">
          <div class="metric-card-top">
            <span class="metric-title">Molecular Weight</span>
            <span class="metric-card-info-icon" title="Click for details">ℹ</span>
          </div>
          <div class="metric-value">${(p.molecular_weight/1000).toFixed(1)}</div>
          <div class="metric-sub">kDa</div>
        </div>
        <div class="metric-card" data-metric="pi" title="Click to view details for Isoelectric Point">
          <div class="metric-card-top">
            <span class="metric-title">Isoelectric Point</span>
            <span class="metric-card-info-icon" title="Click for details">ℹ</span>
          </div>
          <div class="metric-value">${p.isoelectric_point}</div>
          <div class="metric-sub">Theoretical pI</div>
        </div>
        <div class="metric-card" data-metric="instability" title="Click to view details for Instability Index">
          <div class="metric-card-top">
            <span class="metric-title">Instability Index</span>
            <span class="metric-card-info-icon" title="Click for details">ℹ</span>
          </div>
          <div class="metric-value">${p.instability_index}</div>
          <div class="metric-sub">${p.stability_verdict}</div>
        </div>
        <div class="metric-card" data-metric="gravy" title="Click to view details for GRAVY">
          <div class="metric-card-top">
            <span class="metric-title">GRAVY</span>
            <span class="metric-card-info-icon" title="Click for details">ℹ</span>
          </div>
          <div class="metric-value">${p.gravy}</div>
          <div class="metric-sub">Hydropathicity</div>
        </div>
        <div class="metric-card" data-metric="aromaticity" title="Click to view details for Aromaticity">
          <div class="metric-card-top">
            <span class="metric-title">Aromaticity</span>
            <span class="metric-card-info-icon" title="Click for details">ℹ</span>
          </div>
          <div class="metric-value">${p.aromaticity}</div>
          <div class="metric-sub">Phe, Trp, Tyr</div>
        </div>
      </div>

      <div class="metric-click-banner" id="metric-click-banner">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="16" x2="12" y2="12"/>
          <line x1="12" y1="8" x2="12.01" y2="8"/>
        </svg>
        <span><strong>Interactive Biophysical Metrics:</strong> Click any card above to display chemical formulas, reference thresholds, and clinical relevance to Alzheimer's pathology.</span>
      </div>

      <div id="metric-info-display" class="metric-info-display" style="display:none;"></div>

      <div style="background:#ffffff;border:var(--border-light);border-radius:var(--radius-md);padding:1.25rem;">
        <h4 style="font-size:0.9375rem;font-weight:700;margin-bottom:0.75rem;">Amino Acid Composition Frequency</h4>
        <div style="height:240px;position:relative;">
          <canvas id="aa-comp-chart"></canvas>
        </div>
      </div>

      <div class="sequence-viewer-container">
        <div class="seq-viewer-header">
          <h4 style="font-size:0.9375rem;font-weight:700;">Residue Sequence Viewer</h4>
          <div style="display:flex;align-items:center;gap:0.75rem;">
            <button class="btn btn-secondary" id="copy-seq-btn" style="padding:0.35rem 0.75rem;font-size:0.75rem;">Copy Sequence</button>
            <button class="btn btn-secondary" id="download-seq-btn" style="padding:0.35rem 0.75rem;font-size:0.75rem;">Download Analysis</button>
          </div>
        </div>
        <div class="seq-viewer-block" id="formatted-seq-block"></div>
      </div>
    </div>
  `;

  // Draw AA Composition Chart
  setTimeout(() => {
    drawAaCompChart('aa-comp-chart', aaLabels, aaCounts);
    renderFormattedSequence(AppState.currentAnalysis?.input_summary?.sequence || document.getElementById('sequence-textarea')?.value || '');
  }, 50);

  // Copy & Download
  document.getElementById('copy-seq-btn')?.addEventListener('click', () => {
    const raw = document.getElementById('sequence-textarea')?.value || '';
    navigator.clipboard.writeText(raw);
    alert('Protein sequence copied to clipboard.');
  });
  document.getElementById('download-seq-btn')?.addEventListener('click', () => {
    downloadJsonFile(AppState.currentAnalysis?.sequence_properties, `${ident?.gene_name || 'protein'}_sequence_analysis.json`);
  });

  // Wire up interactive metric card clicks
  let activeMetric = null;

  function renderMetricInfo(metricKey) {
    const infoDisplay = document.getElementById('metric-info-display');
    const cards = container.querySelectorAll('.metric-card[data-metric]');
    if (!infoDisplay) return;

    if (activeMetric === metricKey) {
      // Toggle off if clicked again
      activeMetric = null;
      cards.forEach(c => c.classList.remove('active'));
      infoDisplay.style.display = 'none';
      infoDisplay.innerHTML = '';
      return;
    }

    activeMetric = metricKey;
    cards.forEach(c => {
      if (c.getAttribute('data-metric') === metricKey) {
        c.classList.add('active');
      } else {
        c.classList.remove('active');
      }
    });

    const info = BIOPHYS_METRIC_INFO[metricKey];
    if (!info) return;

    infoDisplay.style.display = 'flex';
    infoDisplay.innerHTML = `
      <div class="metric-info-header">
        <div class="metric-info-title-group">
          <span class="metric-info-badge ${info.getBadgeClass(p)}">${info.getDisplayValue(p)}</span>
          <h4 class="metric-info-title">${info.title}</h4>
        </div>
        <button class="metric-info-close-btn" id="close-metric-info" title="Close info">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>
      <div class="metric-info-body">
        <div>
          <div class="metric-info-section-title">Biochemical Definition & Formula</div>
          <p class="metric-info-text">${info.definition}</p>
          <div class="metric-formula-box">
            <code>${info.formulaText}</code>
          </div>
          <div class="metric-info-section-title" style="margin-top:0.85rem;">Sample Protein Assessment</div>
          <div class="metric-info-finding">${info.interpretation(p)}</div>
        </div>
        <div>
          <div class="metric-info-section-title">Standard Reference Thresholds</div>
          <div class="metric-norms-list">
            ${info.referenceNorms.map(n => `
              <div class="metric-norm-row">
                <span class="norm-label">${n.label}</span>
                <span class="norm-range">${n.range}</span>
              </div>
            `).join('')}
          </div>
          <div class="metric-info-section-title" style="margin-top:0.85rem;">Relevance in Alzheimer's Disease Pathology</div>
          <div class="metric-ad-relevance-box">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" style="flex-shrink:0;margin-top:2px;">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="16" x2="12" y2="12"/>
              <line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
            <div>${info.adRelevance}</div>
          </div>
        </div>
      </div>
    `;

    document.getElementById('close-metric-info')?.addEventListener('click', () => {
      activeMetric = null;
      cards.forEach(c => c.classList.remove('active'));
      infoDisplay.style.display = 'none';
      infoDisplay.innerHTML = '';
    });
  }

  container.querySelectorAll('.metric-card[data-metric]').forEach(card => {
    card.addEventListener('click', () => {
      const key = card.getAttribute('data-metric');
      renderMetricInfo(key);
    });
  });
}

function drawAaCompChart(canvasId, labels, data) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return;

  if (AppState.charts['aa']) AppState.charts['aa'].destroy();
  const ctx = canvas.getContext('2d');
  AppState.charts['aa'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Residue Count',
        data: data,
        backgroundColor: '#6366f1',
        borderRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } }
    }
  });
}

function renderFormattedSequence(rawText) {
  const block = document.getElementById('formatted-seq-block');
  if (!block) return;

  const lines = rawText.trim().split('\n');
  let clean = '';
  for (const l of lines) {
    if (!l.trim().startsWith('>')) clean += l.trim().replace(/\s+/g, '');
  }

  clean = clean.toUpperCase();
  let html = '';
  const rowLen = 50;

  for (let i = 0; i < clean.length; i += rowLen) {
    const chunk = clean.slice(i, i + rowLen);
    const startNum = (i + 1).toString().padStart(5, ' ');

    let chunkFormatted = '';
    for (let j = 0; j < chunk.length; j += 10) {
      const block10 = chunk.slice(j, j + 10);
      let coloredBlock = '';
      for (const aa of block10) {
        if (['R','K','H'].includes(aa)) coloredBlock += `<span class="aa-pos">${aa}</span>`;
        else if (['D','E'].includes(aa)) coloredBlock += `<span class="aa-neg">${aa}</span>`;
        else if (['F','W','Y'].includes(aa)) coloredBlock += `<span class="aa-arom">${aa}</span>`;
        else if (['I','L','V','M','A'].includes(aa)) coloredBlock += `<span class="aa-hyd">${aa}</span>`;
        else coloredBlock += aa;
      }
      chunkFormatted += coloredBlock + ' ';
    }

    html += `
      <div class="seq-row">
        <span class="seq-num">${startNum}</span>
        <span class="seq-residues">${chunkFormatted}</span>
      </div>
    `;
  }

  block.innerHTML = html;
}

// 5G. PROTEIN STRUCTURE EXPLORER TAB (3Dmol.js)
function renderStructureViewerTab(struct, ident) {
  const container = document.getElementById('structure-tab-content');
  if (!container) return;

  if (struct.status !== 'available' || !struct.uniprot_accession) {
    container.innerHTML = `
      <div class="empty-state-box">
        <div class="empty-state-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
        </div>
        <h4>No suitable protein structure is currently available.</h4>
        <p style="font-size:0.875rem;">No experimentally resolved (PDB) or predicted (AlphaFold DB) 3D coordinate model is matched to this sequence.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div class="callout-disclaimer" style="margin-bottom:1rem;">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
      <div>
        <strong>Structural Disclaimer:</strong>
        ${struct.caveat}
      </div>
    </div>

    <div class="structure-container">
      <div class="viewer3d-wrapper">
        <div class="viewer3d-controls">
          <select id="mol-style-select" style="font-size:0.75rem;padding:0.25rem;border-radius:4px;border:var(--border-light);">
            <option value="cartoon">Cartoon</option>
            <option value="ribbon">Ribbon</option>
            <option value="stick">Sticks</option>
            <option value="sphere">Spheres</option>
          </select>
          <select id="mol-color-select" style="font-size:0.75rem;padding:0.25rem;border-radius:4px;border:var(--border-light);">
            <option value="spectrum">pLDDT Spectrum</option>
            <option value="chain">By Chain</option>
            <option value="secondary">Secondary Structure</option>
          </select>
          <button class="btn btn-secondary" id="mol-reset-btn" style="padding:0.25rem 0.5rem;font-size:0.75rem;">Reset Camera</button>
          <button class="btn btn-secondary" id="mol-spin-btn" style="padding:0.25rem 0.5rem;font-size:0.75rem;">Spin</button>
        </div>
        <div id="mol3d-viewport"></div>
      </div>
      <div class="structure-sidebar-pane">
        <h4 style="font-size:0.9375rem;font-weight:700;">Model Coordinates</h4>
        <div style="font-size:0.8125rem;display:flex;flex-direction:column;gap:0.5rem;">
          <div><strong>Model ID:</strong> ${struct.entry_id || 'AF-' + struct.uniprot_accession}</div>
          <div><strong>Source:</strong> ${struct.source}</div>
          <div><strong>Accession:</strong> ${struct.uniprot_accession}</div>
          <div><strong>Confidence (pLDDT):</strong> ${struct.mean_plddt ? struct.mean_plddt.toFixed(1) : 'Curated'}</div>
          <div><strong>Status:</strong> ${struct.confidence_category}</div>
        </div>
        <hr style="border:none;border-top:var(--border-light);">
        <a href="${struct.pdb_url}" target="_blank" download class="btn btn-primary" style="font-size:0.8125rem;">Download PDB File</a>
      </div>
    </div>
  `;

  setTimeout(() => {
    init3DmolViewer('mol3d-viewport', struct.uniprot_accession);
  }, 100);
}

function init3DmolViewer(elementId, accession) {
  const elem = document.getElementById(elementId);
  if (!elem || !window.$3Dmol) return;

  elem.innerHTML = '';
  const viewer = $3Dmol.createViewer(elem, { defaultcolors: $3Dmol.rasmolElementColors });
  AppState.molViewer = viewer;

  // Fetch PDB coordinate proxy
  fetch(`/api/proxy-pdb/${accession}`)
    .then(res => {
      if (!res.ok) throw new Error('Could not fetch PDB coordinates');
      return res.text();
    })
    .then(pdbData => {
      viewer.addModel(pdbData, 'pdb');
      viewer.setStyle({}, { cartoon: { color: 'spectrum' } });
      viewer.zoomTo();
      viewer.render();

      // Style select
      document.getElementById('mol-style-select')?.addEventListener('change', (e) => {
        const val = e.target.value;
        const colVal = document.getElementById('mol-color-select')?.value || 'spectrum';
        const colorSpec = colVal === 'spectrum' ? 'spectrum' : (colVal === 'chain' ? 'chain' : 'ssJmol');
        const styleObj = {};
        styleObj[val] = { color: colorSpec };
        viewer.setStyle({}, styleObj);
        viewer.render();
      });

      // Color select
      document.getElementById('mol-color-select')?.addEventListener('change', (e) => {
        const colVal = e.target.value;
        const styleVal = document.getElementById('mol-style-select')?.value || 'cartoon';
        const colorSpec = colVal === 'spectrum' ? 'spectrum' : (colVal === 'chain' ? 'chain' : 'ssJmol');
        const styleObj = {};
        styleObj[styleVal] = { color: colorSpec };
        viewer.setStyle({}, styleObj);
        viewer.render();
      });

      // Reset
      document.getElementById('mol-reset-btn')?.addEventListener('click', () => {
        viewer.zoomTo();
        viewer.render();
      });

      // Spin
      document.getElementById('mol-spin-btn')?.addEventListener('click', () => {
        AppState.isSpinning = !AppState.isSpinning;
        viewer.spin(AppState.isSpinning);
      });
    })
    .catch(err => {
      elem.innerHTML = `<div class="empty-state-box">Failed to render 3D structure: ${err.message}</div>`;
    });
}

// 5H. DETAILED REPORT TAB
function renderDetailedReportTab(data) {
  const container = document.getElementById('detailed-report-content');
  if (!container) return;

  const pred = data.prediction;
  const ident = data.identity;
  const p = data.sequence_properties;

  container.innerHTML = `
    <div class="report-container">
      <div class="report-header-banner">
        <div class="report-meta-header">
          <h2>AD-TRACE Comprehensive Analysis Report</h2>
          <p style="font-size:0.875rem;color:var(--navy-500);margin-top:0.25rem;">
            Generated on ${new Date().toUTCString()} • Research-Use Zero-Leakage Benchmark
          </p>
        </div>
        <div class="report-actions-bar">
          <button class="btn btn-secondary" id="rep-download-csv">Download CSV</button>
          <button class="btn btn-secondary" id="rep-download-json">Download JSON</button>
          <button class="btn btn-primary" id="rep-print-btn">Print Report</button>
        </div>
      </div>

      <div class="report-section">
        <h4 class="report-section-title">1. Input Summary</h4>
        <p style="font-size:0.875rem;">Length: <strong>${data.input_summary.sequence_length} aa</strong> | Validated: <strong>Standard 20 Amino Acids</strong> | FASTA Header: <em>${data.input_summary.fasta_header || 'None'}</em></p>
      </div>

      <div class="report-section">
        <h4 class="report-section-title">2. Protein Identity Resolution</h4>
        <p style="font-size:0.875rem;">
          Best Match: <strong>${ident.best_match || 'Unmatched'}</strong> |
          Accession: <strong>${ident.uniprot_accession || 'N/A'}</strong> |
          Identity: <strong>${ident.sequence_identity ? ident.sequence_identity + '%' : 'N/A'}</strong> |
          Coverage: <strong>${ident.sequence_coverage ? ident.sequence_coverage + '%' : 'N/A'}</strong> |
          Confidence: <em>${ident.match_confidence}</em>
        </p>
      </div>

      <div class="report-section">
        <h4 class="report-section-title">3. Model Classification Result</h4>
        <p style="font-size:0.875rem;">
          Classification: <strong>${pred.label}</strong> |
          Model Probability: <strong>${pred.probability}</strong> |
          Decision Threshold: <strong>${pred.threshold}</strong> |
          Architecture: <strong>${pred.model_architecture}</strong>
        </p>
      </div>

      <div class="report-section">
        <h4 class="report-section-title">4. Statistical Limitations & Zero-Leakage Protocol</h4>
        <ul style="font-size:0.8125rem;padding-left:1.25rem;line-height:1.6;color:var(--navy-700);">
          ${pred.important_limitations.map(l => `<li>${l}</li>`).join('')}
        </ul>
      </div>

      <div class="report-section">
        <h4 class="report-section-title">5. Biophysical Sequence Properties</h4>
        <p style="font-size:0.875rem;">
          Molecular Weight: <strong>${p.molecular_weight} Da</strong> |
          Isoelectric Point (pI): <strong>${p.isoelectric_point}</strong> |
          GRAVY: <strong>${p.gravy}</strong> |
          Instability: <strong>${p.instability_index} (${p.stability_verdict})</strong>
        </p>
      </div>

      <div class="report-section">
        <h4 class="report-section-title">6. Brain Expression Evidence (GTEx v10)</h4>
        <p style="font-size:0.875rem;">
          Status: <strong>${data.brain_expression.status}</strong> |
          Expression Type: <em>${data.brain_expression.expression_type || 'N/A'}</em>
        </p>
      </div>

      <div class="report-section">
        <h4 class="report-section-title">7. Interaction Network Evidence (STRING v12)</h4>
        <p style="font-size:0.875rem;">
          Status: <strong>${data.interactions.status}</strong> |
          Interactors Found: <strong>${data.interactions.total_interactors || 0}</strong>
        </p>
      </div>

      <div class="report-section">
        <h4 class="report-section-title">8. Tertiary Structure Coordinates (AlphaFold DB)</h4>
        <p style="font-size:0.875rem;">
          Status: <strong>${data.structure.status}</strong> |
          Model: <em>${data.structure.entry_id || 'N/A'}</em>
        </p>
      </div>

      <div class="report-section">
        <h4 class="report-section-title">9. Provenance & Scientific Citations</h4>
        <ul style="font-size:0.8125rem;padding-left:1.25rem;line-height:1.6;color:var(--navy-600);">
          ${data.evidence_sources.map(s => `<li><strong>${s.database}:</strong> ${s.citation}</li>`).join('')}
        </ul>
      </div>
    </div>
  `;

  // Attach download handlers
  document.getElementById('rep-download-csv')?.addEventListener('click', () => {
    downloadReportCsv(data);
  });
  document.getElementById('rep-download-json')?.addEventListener('click', () => {
    downloadJsonFile(data, `${ident.gene_name || 'analysis'}_report.json`);
  });
  document.getElementById('rep-print-btn')?.addEventListener('click', () => {
    window.print();
  });
}

function downloadJsonFile(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadReportCsv(data) {
  const rows = [
    ['Metric', 'Value'],
    ['Input Length', data.input_summary.sequence_length],
    ['Predicted Label', data.prediction.label],
    ['Model Probability', data.prediction.probability],
    ['Decision Threshold', data.prediction.threshold],
    ['Resolved Gene', data.identity.gene_name || 'None'],
    ['UniProt Accession', data.identity.uniprot_accession || 'None'],
    ['Sequence Identity', data.identity.sequence_identity || 'None'],
    ['Molecular Weight (Da)', data.sequence_properties.molecular_weight],
    ['Isoelectric Point (pI)', data.sequence_properties.isoelectric_point],
    ['Hydropathy (GRAVY)', data.sequence_properties.gravy]
  ];
  const csvContent = rows.map(e => e.join(',')).join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${data.identity.gene_name || 'protein'}_report.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ========================================================
// 6. TABS NAVIGATION
// ========================================================
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tabId = btn.dataset.tab;
      AppState.activeTab = tabId;

      const panels = document.querySelectorAll('.tab-panel');
      panels.forEach(p => p.classList.remove('active'));
      const activePanel = document.getElementById(`tab-panel-${tabId}`);
      if (activePanel) activePanel.classList.add('active');
    });
  });
}

// ========================================================
// 7. MODALS
// ========================================================
function initModals() {
  // Close modals on close button click

  // Close modals on close button click
  document.querySelectorAll('.modal-close-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.modal-backdrop').forEach(m => m.classList.remove('open'));
    });
  });

  // Close modals on backdrop click
  document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) {
        backdrop.classList.remove('open');
      }
    });
  });
}

function openModal(modalId) {
  const m = document.getElementById(modalId);
  if (m) m.classList.add('open');
}

window.openDocsFromModal = function() {
  document.querySelectorAll('.modal-backdrop').forEach(m => m.classList.remove('open'));
  const navItem = document.getElementById('nav-item-methodology');
  if (navItem) {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    navItem.classList.add('active');
    switchView('methodology');
    const scrollContainer = document.querySelector('.workspace-scroll') || window;
    scrollContainer.scrollTo({ top: 0, behavior: 'smooth' });
  }
};

window.openBenchmarksFromModal = function() {
  document.querySelectorAll('.modal-backdrop').forEach(m => m.classList.remove('open'));
  const navItem = document.getElementById('nav-item-benchmarks');
  if (navItem) {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    navItem.classList.add('active');
    switchView('benchmarks');
    const scrollContainer = document.querySelector('.workspace-scroll') || window;
    scrollContainer.scrollTo({ top: 0, behavior: 'smooth' });
  }
};

function openAlignmentModal(ident) {
  const modal = document.getElementById('alignment-modal');
  const body = document.getElementById('alignment-modal-body');
  if (!modal || !body || !ident.alignment) return;

  const a = ident.alignment;
  body.innerHTML = `
    <div style="font-size:0.875rem;line-height:1.6;">
      <p><strong>Query against target:</strong> ${ident.gene_name} (${ident.uniprot_accession})</p>
      <p><strong>Matching Residues:</strong> ${a.matches} / ${a.length} (${ident.sequence_identity}%)</p>
      <div style="margin-top:1rem;background:var(--bg-app);padding:1rem;border-radius:var(--radius-sm);font-family:var(--font-mono);font-size:0.8125rem;">
        <div><strong>Query :</strong> ${a.snippet_query}</div>
        <div style="color:var(--navy-400);margin:2px 0;">       ||||||||||||||||||||||||||||||</div>
        <div><strong>Target:</strong> ${a.snippet_target}</div>
      </div>
    </div>
  `;
  modal.classList.add('open');
}

// Render Examples Catalog View
function renderExamplesView(examples) {
  const container = document.getElementById('examples-catalog-grid');
  if (!container) return;

  container.innerHTML = examples.map(e => `
    <div class="card" style="padding:1.25rem;display:flex;flex-direction:column;gap:0.75rem;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <h4 style="font-size:1.125rem;font-weight:700;color:var(--navy-900);">${e.gene}</h4>
          <span style="font-size:0.8125rem;color:var(--navy-500);">${e.protein_name}</span>
        </div>
        <span class="pill-exact">${e.accession}</span>
      </div>
      <p style="font-size:0.8125rem;color:var(--navy-700);line-height:1.5;">${e.description}</p>
      <div style="font-size:0.75rem;color:var(--navy-400);">Length: ${e.sequence_length} aa | MW: ${(e.molecular_weight/1000).toFixed(1)} kDa</div>
      <button class="btn btn-primary" onclick="loadAndAnalyzeExample('${e.gene}')" style="margin-top:auto;font-size:0.8125rem;">
        Load into Workspace
      </button>
    </div>
  `).join('');
}

window.loadAndAnalyzeExample = function(gene) {
  document.querySelector('[data-view="analyze"]')?.click();
  loadExampleByGene(gene);
};

// ========================================================
// 8. LIGHTBOX MODAL FOR PUBLICATION GRAPHS
// ========================================================
window.openLightbox = function(src, caption) {
  const modal = document.getElementById('graph-lightbox-modal');
  const img = document.getElementById('lightbox-img');
  const cap = document.getElementById('lightbox-caption');
  if (modal && img) {
    img.src = src;
    if (cap) cap.textContent = caption || '';
    modal.classList.add('open');
  }
};

window.closeLightbox = function() {
  const modal = document.getElementById('graph-lightbox-modal');
  if (modal) modal.classList.remove('open');
};

// Lightbox dismiss on backdrop click and ESC key
document.addEventListener('DOMContentLoaded', () => {
  const lbModal = document.getElementById('graph-lightbox-modal');
  lbModal?.addEventListener('click', (e) => {
    if (e.target === lbModal) {
      closeLightbox();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeLightbox();
      document.querySelectorAll('.modal-backdrop').forEach(m => m.classList.remove('open'));
    }
  });
});
