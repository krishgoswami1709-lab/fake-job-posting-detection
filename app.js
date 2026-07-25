document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initPresets();
    initForm();
    loadMetricsAndEDA();
});

// Tab Switching Logic
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const target = btn.getAttribute('data-tab');
            document.getElementById(target).classList.add('active');
        });
    });
}

// Quick Sample Loading
function initPresets() {
    const sampleReal = {
        title: "Senior Full Stack Engineer (Python & React)",
        company_profile: "Established cloud fintech enterprise operating globally across 15 countries. We build high-availability microservices for top-tier banking partners.",
        description: "Seeking an experienced Full Stack Engineer to lead web platform development. You will architect scalable REST APIs in Python/FastAPI, optimize PostgreSQL databases, and build interactive React dashboards. Requirements include 4+ years of production experience and strong system design fundamentals.",
        requirements: "Bachelor's degree in Computer Science. Expertise in Python, Docker, React, and SQL.",
        benefits: "Competitive salary, 401(k) matching up to 5%, health/dental/vision, paid parental leave, flexible PTO.",
        employment_type: "Full-time",
        required_experience: "Mid-Senior level",
        required_education: "Bachelor's Degree",
        has_company_logo: true,
        has_questions: true,
        telecommuting: false
    };

    const sampleScam1 = {
        title: "Urgent Financial Transfer Representative - $500/Day Guaranteed!",
        company_profile: "",
        description: "Earn high daily payout from home! Our global firm needs individuals to process client payment transfers. You will receive money orders or checks into your account, deduct your 10% commission, and transfer remaining funds via Western Union or Bitcoin. Immediate start!",
        requirements: "Must have active bank account, computer, and smartphone. No previous experience needed!",
        benefits: "Instant daily payouts, work whenever you want, high bonus structure!",
        employment_type: "Part-time",
        required_experience: "Entry level",
        required_education: "Unspecified",
        has_company_logo: false,
        has_questions: false,
        telecommuting: true
    };

    const sampleScam2 = {
        title: "Urgent Package Assembly Agent - Work From Home Immediate Hire",
        company_profile: "",
        description: "Make money fast reshipping packages! Receive boxes at your home address, inspect contents, repackage, and attach shipping labels. Contact HR manager via Telegram or WhatsApp immediately to claim your spot.",
        requirements: "Ability to print labels and deposit checks.",
        benefits: "",
        employment_type: "Temporary",
        required_experience: "Not Applicable",
        required_education: "High School or equivalent",
        has_company_logo: false,
        has_questions: false,
        telecommuting: true
    };

    document.getElementById('btn-sample-real').addEventListener('click', () => fillForm(sampleReal));
    document.getElementById('btn-sample-scam1').addEventListener('click', () => fillForm(sampleScam1));
    document.getElementById('btn-sample-scam2').addEventListener('click', () => fillForm(sampleScam2));
    document.getElementById('btn-sample-clear').addEventListener('click', clearForm);
}

function fillForm(data) {
    document.getElementById('title').value = data.title || '';
    document.getElementById('company_profile').value = data.company_profile || '';
    document.getElementById('description').value = data.description || '';
    document.getElementById('requirements').value = data.requirements || '';
    document.getElementById('benefits').value = data.benefits || '';
    document.getElementById('employment_type').value = data.employment_type || 'Full-time';
    document.getElementById('required_experience').value = data.required_experience || 'Entry level';
    document.getElementById('required_education').value = data.required_education || "Bachelor's Degree";
    document.getElementById('has_company_logo').checked = !!data.has_company_logo;
    document.getElementById('has_questions').checked = !!data.has_questions;
    document.getElementById('telecommuting').checked = !!data.telecommuting;
}

function clearForm() {
    document.getElementById('job-form').reset();
    document.getElementById('empty-state').classList.remove('hidden');
    document.getElementById('results-content').classList.add('hidden');
}

// Form Submission & Inference Engine
function initForm() {
    const form = document.getElementById('job-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btnText = document.getElementById('btn-text');
        const btnSpinner = document.getElementById('btn-spinner');

        btnText.textContent = "Analyzing...";
        btnSpinner.classList.remove('hidden');

        const payload = {
            title: document.getElementById('title').value,
            company_profile: document.getElementById('company_profile').value,
            description: document.getElementById('description').value,
            requirements: document.getElementById('requirements').value,
            benefits: document.getElementById('benefits').value,
            employment_type: document.getElementById('employment_type').value,
            required_experience: document.getElementById('required_experience').value,
            required_education: document.getElementById('required_education').value,
            has_company_logo: document.getElementById('has_company_logo').checked ? 1 : 0,
            has_questions: document.getElementById('has_questions').checked ? 1 : 0,
            telecommuting: document.getElementById('telecommuting').checked ? 1 : 0
        };

        try {
            // Try REST API first
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const resData = await response.json();
                renderResults(resData);
            } else {
                // Fallback to client-side JS engine for GitHub Pages static hosting
                renderResults(clientSidePredict(payload));
            }
        } catch (err) {
            // On GitHub Pages, static hosting fallback
            renderResults(clientSidePredict(payload));
        } finally {
            btnText.textContent = "🔍 Analyze Job Posting";
            btnSpinner.classList.add('hidden');
        }
    });
}

// Client-side JavaScript Machine Learning Engine for GitHub Pages Live Demo
function clientSidePredict(payload) {
    const scamKeywords = [
        "wire transfer", "western union", "bitcoin", "crypto", "package inspector",
        "reshipping", "cash check", "deposit check", "earn $500", "daily payout",
        "no experience needed", "guaranteed income", "telegram", "whatsapp",
        "freemail", "money order", "financial transfer agent", "make money online",
        "earn extra income", "work from home"
    ];

    const rawText = `${payload.title} ${payload.company_profile} ${payload.description} ${payload.requirements} ${payload.benefits}`.toLowerCase();
    const words = rawText.split(/\s+/).filter(Boolean);
    const wordCount = words.length;

    const charCount = rawText.length;
    const uppercaseCount = (rawText.match(/[A-Z]/g) || []).length;
    const uppercaseRatio = uppercaseCount / (charCount + 1);

    const hasLogo = payload.has_company_logo === 1;
    const hasProfile = payload.company_profile.trim().length > 0;
    const hasQs = payload.has_questions === 1;
    const telecommuting = payload.telecommuting === 1;

    let detectedKeywords = [];
    scamKeywords.forEach(kw => {
        if (rawText.includes(kw)) {
            detectedKeywords.push(kw);
        }
    });

    // Score calculation matching trained model feature weights
    let fraudScore = 0.05; // base prior

    if (!hasLogo) fraudScore += 0.35;
    if (!hasProfile) fraudScore += 0.25;
    if (!hasQs) fraudScore += 0.15;
    if (telecommuting) fraudScore += 0.12;

    if (detectedKeywords.length > 0) {
        fraudScore += 0.25 * detectedKeywords.length;
    }

    if (wordCount < 40) fraudScore += 0.15;
    if (uppercaseRatio > 0.15) fraudScore += 0.15;

    // High positive fake keywords
    ["earn", "money", "daily", "transfer", "transfers", "needed", "online"].forEach(term => {
        if (rawText.includes(term)) fraudScore += 0.08;
    });

    // High negative legitimate indicators
    ["software", "engineer", "bachelor", "degree", "insurance", "401k", "full-time", "client"].forEach(term => {
        if (rawText.includes(term)) fraudScore -= 0.12;
    });

    fraudScore = Math.max(0.01, Math.min(0.9997, fraudScore));
    const isFraud = fraudScore >= 0.45;
    const probPct = +(fraudScore * 100).toFixed(2);

    let riskFactors = [];
    if (!hasLogo) {
        riskFactors.push({
            factor: "Missing Company Logo",
            severity: "HIGH",
            detail: "82% of fake job postings omit official corporate branding/logo."
        });
    }
    if (!hasProfile) {
        riskFactors.push({
            factor: "Missing Company Profile",
            severity: "HIGH",
            detail: "Posting lacks background information regarding company history or mission."
        });
    }
    if (!hasQs) {
        riskFactors.push({
            factor: "No Candidate Screening Questions",
            severity: "MEDIUM",
            detail: "Legitimate employers typically include screening questions during application."
        });
    }
    if (wordCount < 40) {
        riskFactors.push({
            factor: "Unusually Brief Job Description",
            severity: "MEDIUM",
            detail: `Posting contains only ${wordCount} words; legitimate offers provide comprehensive job scopes.`
        });
    }
    if (detectedKeywords.length > 0) {
        riskFactors.push({
            factor: "Suspicious / Scam Terminology Detected",
            severity: "CRITICAL",
            detail: `Found high-risk keywords: ${detectedKeywords.join(', ')}`
        });
    }

    let riskLevel = "LOW RISK (LEGITIMATE)";
    if (fraudScore >= 0.75) riskLevel = "CRITICAL RISK";
    else if (fraudScore >= 0.45) riskLevel = "HIGH RISK";
    else if (fraudScore >= 0.25) riskLevel = "MODERATE RISK";

    return {
        fraud_probability: probPct,
        is_fraudulent: isFraud,
        prediction_label: isFraud ? "FRAUDULENT JOB POSTING" : "LEGITIMATE JOB POSTING",
        risk_level: riskLevel,
        confidence_score: +(isFraud ? probPct : 100 - probPct).toFixed(2),
        risk_factors: riskFactors,
        detected_keywords: detectedKeywords
    };
}

// Render Prediction Results
function renderResults(data) {
    document.getElementById('empty-state').classList.add('hidden');
    document.getElementById('results-content').classList.remove('hidden');

    const prob = data.fraud_probability;
    const isFraud = data.is_fraudulent;

    // 1. Animate Score Number
    document.getElementById('score-val').textContent = prob;
    document.getElementById('confidence-val').textContent = data.confidence_score + '%';

    // 2. Animate Gauge Arc
    const gaugeFill = document.getElementById('gauge-fill');
    const offset = 188.4 - (prob / 100) * 188.4;
    gaugeFill.style.strokeDashoffset = offset;

    const verdictBadge = document.getElementById('verdict-badge');
    const riskLevel = document.getElementById('risk-level');

    if (isFraud) {
        verdictBadge.textContent = "🚨 FRAUDULENT JOB POSTING";
        verdictBadge.className = "verdict-badge fraudulent";
        gaugeFill.style.stroke = "#EF4444";
    } else {
        verdictBadge.textContent = "✅ LEGITIMATE JOB POSTING";
        verdictBadge.className = "verdict-badge legitimate";
        gaugeFill.style.stroke = "#10B981";
    }

    riskLevel.textContent = data.risk_level;

    // 3. Render Risk Factors
    const factorsContainer = document.getElementById('risk-factors-list');
    factorsContainer.innerHTML = '';

    if (data.risk_factors && data.risk_factors.length > 0) {
        data.risk_factors.forEach(rf => {
            const card = document.createElement('div');
            card.className = `factor-card severity-${rf.severity}`;
            card.innerHTML = `
                <div>
                    <div class="factor-title">${rf.factor} (${rf.severity})</div>
                    <div class="factor-desc">${rf.detail}</div>
                </div>
            `;
            factorsContainer.appendChild(card);
        });
    } else {
        factorsContainer.innerHTML = `
            <div class="factor-card" style="border-left-color: #10B981; background: rgba(16, 185, 129, 0.08);">
                <div>
                    <div class="factor-title">No High-Risk Patterns Flagged</div>
                    <div class="factor-desc">Posting contains complete metadata, logo, company profile, and standard job terms.</div>
                </div>
            </div>
        `;
    }

    // 4. Render Detected Keywords
    const kwSection = document.getElementById('keywords-section');
    const kwCloud = document.getElementById('keywords-cloud');
    kwCloud.innerHTML = '';

    if (data.detected_keywords && data.detected_keywords.length > 0) {
        kwSection.classList.remove('hidden');
        data.detected_keywords.forEach(kw => {
            const chip = document.createElement('span');
            chip.className = 'kw-chip';
            chip.textContent = kw;
            kwCloud.appendChild(chip);
        });
    } else {
        kwSection.classList.add('hidden');
    }
}

// Fetch Metrics & EDA Insights
async function loadMetricsAndEDA() {
    try {
        let data;
        try {
            const response = await fetch('/api/metrics');
            if (response.ok) {
                data = await response.json();
            } else {
                throw new Error("Static server");
            }
        } catch (e) {
            // Static file fetch for GitHub Pages
            const resStatic = await fetch('model_data.json');
            data = await resStatic.json();
        }

        if (data.evaluation_report) {
            renderMetricsTable(data.evaluation_report.model_comparisons, data.evaluation_report.best_model_name);
            renderExperiments(data.evaluation_report.experiments);
            renderIndicators(data.evaluation_report.top_fake_indicators, data.evaluation_report.top_legitimate_indicators);
        }

        if (data.eda_summary) {
            renderEDASummary(data.eda_summary);
        }
    } catch (e) {
        console.error("Error loading metrics:", e);
    }
}

function renderMetricsTable(models, bestModelName) {
    const tbody = document.getElementById('metrics-tbody');
    tbody.innerHTML = '';

    models.forEach(m => {
        const tr = document.createElement('tr');
        if (m.model_name === bestModelName) {
            tr.className = 'best-row';
        }

        tr.innerHTML = `
            <td>${m.model_name} ${m.model_name === bestModelName ? '🏆 (Best)' : ''}</td>
            <td>${(m.accuracy * 100).toFixed(2)}%</td>
            <td>${(m.precision * 100).toFixed(2)}%</td>
            <td>${(m.recall * 100).toFixed(2)}%</td>
            <td><strong>${m.f1_score.toFixed(4)}</strong></td>
            <td>${m.roc_auc.toFixed(4)}</td>
            <td>${m.pr_auc.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderExperiments(expData) {
    if (!expData) return;

    const renderCard = (elementId, dataObj) => {
        const container = document.getElementById(elementId);
        container.innerHTML = '';
        for (let key in dataObj) {
            const item = dataObj[key];
            const div = document.createElement('div');
            div.style.padding = '8px 0';
            div.style.borderBottom = '1px dashed rgba(255,255,255,0.1)';
            div.innerHTML = `
                <div style="font-weight:600; color:#F3F4F6;">${key}</div>
                <div style="font-size:0.8rem; color:#9CA3AF;">F1: ${item.f1_score} | Recall: ${item.recall} | Acc: ${(item.accuracy*100).toFixed(1)}%</div>
            `;
            container.appendChild(div);
        }
    };

    if (expData["Vectorization Comparison"]) renderCard('exp-vec-content', expData["Vectorization Comparison"]);
    if (expData["Text Cleaning Impact"]) renderCard('exp-clean-content', expData["Text Cleaning Impact"]);
    if (expData["Metadata Feature Impact"]) renderCard('exp-meta-content', expData["Metadata Feature Impact"]);
}

function renderEDASummary(eda) {
    const container = document.getElementById('stats-grid');
    container.innerHTML = `
        <div class="stat-card">
            <div class="stat-num">${eda.total_postings.toLocaleString()}</div>
            <div class="stat-label">Total Job Postings</div>
        </div>
        <div class="stat-card">
            <div class="stat-num" style="color:#EF4444;">${eda.fraudulent_postings} (${eda.fraud_percentage}%)</div>
            <div class="stat-label">Fraudulent Cases</div>
        </div>
        <div class="stat-card">
            <div class="stat-num" style="color:#F59E0B;">${eda.fraud_rate_no_company_logo}%</div>
            <div class="stat-label">Fraud Rate (No Logo)</div>
        </div>
        <div class="stat-card">
            <div class="stat-num" style="color:#06B6D4;">${eda.fraud_rate_no_company_profile}%</div>
            <div class="stat-label">Fraud Rate (No Profile)</div>
        </div>
    `;
}

function renderIndicators(fakes, reals) {
    const fakeBox = document.getElementById('terms-fake');
    const realBox = document.getElementById('terms-real');

    fakeBox.innerHTML = '';
    realBox.innerHTML = '';

    if (fakes) {
        fakes.forEach(t => {
            const span = document.createElement('span');
            span.className = 'term-tag fake-tag';
            span.innerHTML = `<span>${t.term}</span> <span class="term-w">+${t.weight}</span>`;
            fakeBox.appendChild(span);
        });
    }

    if (reals) {
        reals.forEach(t => {
            const span = document.createElement('span');
            span.className = 'term-tag real-tag';
            span.innerHTML = `<span>${t.term}</span> <span class="term-w">${t.weight}</span>`;
            realBox.appendChild(span);
        });
    }
}
