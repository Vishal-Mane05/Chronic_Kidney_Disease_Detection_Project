/**
 * CKD Predictor - Main JavaScript File
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. GLOBAL LOADER LOGIC
    const loader = document.getElementById("loading-screen");
    if (loader) loader.style.display = "none";

    const forms = document.querySelectorAll("form");
    forms.forEach(form => {
        form.addEventListener("submit", function () {
            if (loader) loader.style.display = "flex";
        });
    });

    // 2. CT SCAN SLIDER LOGIC
    const slider = document.getElementById("slider");
    const overlay = document.getElementById("overlay");
    if (slider && overlay) {
        slider.addEventListener("input", function(e) {
            overlay.style.width = e.target.value + "%";
        });
    }

    // 3. CONFIDENCE BAR ANIMATION
    const bar = document.getElementById("confidence-fill");
    const text = document.getElementById("confidence-text");
    if (bar && text) {
        const value = parseFloat(bar.getAttribute("data-value"));
        let width = 0;
        const interval = setInterval(() => {
            if (width >= value) clearInterval(interval);
            else {
                width++;
                bar.style.width = width + "%";
                text.innerText = width + "%";
            }
        }, 10);
    }

    // 4. CT SCAN UPLOAD: REMOVE LOGIC
    const removeBtnEl = document.getElementById("remove-btn");
    if(removeBtnEl) {
        removeBtnEl.addEventListener("click", function () {
            document.getElementById('preview').style.display = "none";
            document.getElementById('upload-content').style.display = "block";
            document.getElementById('remove-btn').style.display = "none";
            document.getElementById('upload-box').classList.remove("active");
            document.getElementById('analyze-btn').disabled = true;
            document.getElementById('file').value = ""; 
        });
    }
});

// Safari/Mobile Back-Button fix for Loader
window.addEventListener("pageshow", function () {
    const loader = document.getElementById("loading-screen");
    if (loader) loader.style.display = "none";
});

// CT Scan Upload Preview
function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const preview = document.getElementById('preview');
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        document.getElementById('upload-content').style.display = "none";
        document.getElementById('remove-btn').style.display = "inline-block";
        document.getElementById('upload-box').classList.add("active");
        document.getElementById('analyze-btn').disabled = false;
    }
}

// Dev Tools: Auto-Fill
function fillCKD() {
    const fields = {
        "age": 65, "bloodpressure": 90, "specificgravity": 1.010, "albumin": 4, 
        "sugar": 2, "bloodglucoserandom": 200, "bloodurea": 100, "serumcreatinine": 4.0, 
        "sodium": 130, "potassium": 6.0, "hemoglobin": 8, "packedellvolume": 25, 
        "wbccount": 15000, "rbccount": 3.0, "redbloodcells": "abnormal", "puscells": "abnormal", 
        "puscellsclumps": "present", "bacteriapresence": "present", "hypertension": "yes", 
        "diabetesmellitus": "yes", "coronaryarterydisease": "yes", "appetite": "poor", 
        "pedaledema": "yes", "anemia": "yes"
    };
    for (const [key, value] of Object.entries(fields)) {
        const el = document.getElementsByName(key)[0];
        if (el) el.value = value;
    }
}

function fillNormal() {
    const fields = {
        "age": 30, "bloodpressure": 80, "specificgravity": 1.020, "albumin": 0, 
        "sugar": 0, "bloodglucoserandom": 100, "bloodurea": 30, "serumcreatinine": 1.0, 
        "sodium": 140, "potassium": 4.5, "hemoglobin": 15, "packedellvolume": 44, 
        "wbccount": 8000, "rbccount": 5.0, "redbloodcells": "normal", "puscells": "normal", 
        "puscellsclumps": "notpresent", "bacteriapresence": "notpresent", "hypertension": "no", 
        "diabetesmellitus": "no", "coronaryarterydisease": "no", "appetite": "good", 
        "pedaledema": "no", "anemia": "no"
    };
    for (const [key, value] of Object.entries(fields)) {
        const el = document.getElementsByName(key)[0];
        if (el) el.value = value;
    }
}

// 8. PDF REPORT GENERATOR
function downloadPDF() {
    const element = document.querySelector('.result-card');
    const buttons = document.getElementById('action-buttons');
    
    if (element && buttons) {
        buttons.style.display = 'none';
        element.classList.add('pdf-format');

        const dateSpan = document.getElementById('report-date');
        if (dateSpan) dateSpan.innerText = new Date().toLocaleDateString();

        const opt = {
            margin:       0.5,
            filename:     'CKD_Patient_Report.pdf',
            image:        { type: 'jpeg', quality: 1.0 },
            html2canvas:  { scale: 2, useCORS: true, letterRendering: true },
            jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
        };

        html2pdf().set(opt).from(element).save().then(() => {
            buttons.style.display = 'block';
            element.classList.remove('pdf-format');
        });
    }
}