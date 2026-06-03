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

function handleFileSelection(file) {

    if (!file.type.startsWith('image/')) {
        alert("Selecciona una imagen válida");
        return;
    }

    currentFile = file;

    const reader = new FileReader();

    reader.onload = function () {

        previewImage.src = reader.result;

        previewContainer.classList.remove(
            'hidden'
        );
    };

    reader.readAsDataURL(file);
}

imageInput.addEventListener(
    'change',
    function () {

        if (this.files[0]) {
            handleFileSelection(
                this.files[0]
            );
        }
    }
);

window.addEventListener(
    'paste',
    function (event) {

        const items =
            event.clipboardData.items;

        for (let i = 0; i < items.length; i++) {

            if (
                items[i].type.indexOf(
                    'image'
                ) !== -1
            ) {

                const file =
                    items[i].getAsFile();

                handleFileSelection(file);

                break;
            }
        }
    }
);

analyzeBtn.addEventListener(
    'click',
    async () => {

        if (!currentFile) {

            alert(
                "Selecciona una imagen primero"
            );

            return;
        }

        spinner.classList.remove(
            'hidden'
        );

        btnText.innerText =
            "Analizando...";

        analyzeBtn.disabled = true;

        resultContainer.classList.add(
            'hidden'
        );

        const formData =
            new FormData();

        formData.append(
            'file',
            currentFile
        );

        try {

            const response =
                await fetch(
                    '/predict',
                    {
                        method: 'POST',
                        body: formData
                    }
                );

            if (!response.ok) {
                throw new Error(
                    "Error del servidor"
                );
            }

            const data =
                await response.json();

            detectionLabel.innerText =
                data.label;

            confidenceLabel.innerText =
                (
                    data.confidence * 100
                ).toFixed(2) + "%";

            resultContainer.classList.remove(
                'hidden'
            );

            if (
                data.confidence > 0.70
            ) {

                confetti({
                    particleCount: 150,
                    spread: 80,
                    origin: {
                        y: 0.6
                    }
                });
            }

        } catch (error) {

            console.error(error);

            alert(
                "Error conectando con la IA"
            );

        } finally {

            spinner.classList.add(
                'hidden'
            );

            btnText.innerText =
                "Analizar Imagen";

            analyzeBtn.disabled = false;
        }
    }
);