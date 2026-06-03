const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const previewContainer = document.getElementById('previewContainer');
const analyzeBtn = document.getElementById('analyzeBtn');
const spinner = document.getElementById('spinner');
const btnText = document.getElementById('btnText');
const resultContainer = document.getElementById('resultContainer');
const detectionLabel = document.getElementById('detectionLabel');
const confidenceLabel = document.getElementById('confidenceLabel');

let currentFile = null;

// Controlar y desplegar la previsualización de la foto
function handleFileSelection(file) {
    if (file && file.type.startsWith('image/')) {
        currentFile = file;
        const reader = new FileReader();
        
        reader.addEventListener('load', function() {
            previewImage.setAttribute('src', this.result);
            previewContainer.classList.remove('hidden'); 
        });
        
        reader.readAsDataURL(file);
    } else {
        alert("Por favor, sube un formato de imagen válido (.jpg, .png).");
    }
}

// Carga por explorador de archivos tradicional
imageInput.addEventListener('change', function() {
    if (this.files[0]) {
        handleFileSelection(this.files[0]);
    }
});

// Capturar pegado desde el portapapeles (Ctrl + V)
window.addEventListener('paste', function(event) {
    const items = (event.clipboardData || event.originalEvent.clipboardData).items;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
            const file = items[i].getAsFile();
            handleFileSelection(file);
            break;
        }
    }
});

// Petición de inferencia a FastAPI
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) {
        alert("Primero debes seleccionar o pegar una foto de un vehículo.");
        return;
    }

    // Activar spinner de carga en el botón
    spinner.classList.remove('hidden');
    btnText.innerText = "Analizando...";
    analyzeBtn.disabled = true;
    resultContainer.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Fallo en comunicación");

        const data = await response.json();

        // Desplegar resultados e iluminar etiquetas
        detectionLabel.innerText = data.label;
        confidenceLabel.innerText = (data.confidence * 100).toFixed(2) + "%";
        resultContainer.classList.remove('hidden');

        // 🎉 Si el modelo tiene buena confianza, lanza confeti
        if (data.confidence > 0.70) {
            confetti({
                particleCount: 140,
                spread: 75,
                origin: { y: 0.65 }
            });
        }

    } catch (error) {
        console.error(error);
        alert("No se pudo obtener respuesta del servidor local de IA.");
    } finally {
        // Restaurar estado del botón
        spinner.classList.add('hidden');
        btnText.innerText = "Analizar Imagen";
        analyzeBtn.disabled = false;
    }
});