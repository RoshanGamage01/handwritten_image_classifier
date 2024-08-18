const canvas = document.getElementById('canvas');
const submit_button = document.getElementById('submit');
const random_button = document.getElementById('random');
const result = document.getElementById('result');
const clear_button = document.getElementById('clear');
const probabilityBars = document.getElementsByClassName("prob");

const ctx = canvas.getContext('2d');

const width = canvas.width
const height = canvas.height
const pixelSize = 10;
const gridSize = 28;

let debounceTimer;
let lastCall = 0;
let drawing = false;
let lastPosition = null;

class Pixel {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.grayValue = 0
    }

    draw() {
        ctx.strokeStyle = '#ddd';
        ctx.strokeRect(this.x * pixelSize, this.y * pixelSize, pixelSize, pixelSize);
    }

    updateColor(grayValue) {
        this.grayValue = Math.min(1, this.grayValue + grayValue);

        ctx.fillStyle = `rgba(0, 0, 0, ${this.grayValue})`;
        ctx.fillRect(this.x * pixelSize, this.y * pixelSize, pixelSize, pixelSize);
        throttleSubmit();
    }
}

const pixels = [];

// Load pixels
for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
        pixels.push(new Pixel(i, j));
    }
}

// Draw pixel grid
function draw_pixel_grid() {
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            pixels[i * gridSize + j].draw();
            pixels[i * gridSize + j].grayValue = 0;
        }
    }
}

draw_pixel_grid();


// Fill pixels according to mouse position and brush size
function fill_pixel(x, y) {
    const index = x * gridSize + y;

    const brushSize = 1.3; // 1.5 is good 
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const distance = Math.sqrt((x - i) ** 2 + (y - j) ** 2);
            const grayValue = 1-distance / brushSize;
            
            if (grayValue > 0) {
                pixels[i * gridSize + j].updateColor(grayValue);
            }
        }
    }
}


canvas.addEventListener('mousedown', (event) => {
    drawing = true;
    const { x, y } = getMousePosition(event);
    fill_pixel(x, y);
    lastPosition = { x, y };
});

canvas.addEventListener('mousemove', (event) => {
    if (drawing) {
        const { x, y } = getMousePosition(event);
        // fill_pixel(x, y);
        if (lastPosition) {
            drawLine(lastPosition.x, lastPosition.y, x, y);
        }
        lastPosition = { x, y };
    }
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
    astPosition = null;
});

canvas.addEventListener('mouseleave', () => {
    drawing = false;
    astPosition = null;
});

function getMousePosition(event) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / pixelSize);
    const y = Math.floor((event.clientY - rect.top) / pixelSize);
    
    return { x, y };
}

// Bresenham's line algorithm
// https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
function drawLine(x0, y0, x1, y1) {
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;

    while (true) {
        fill_pixel(x0, y0);

        if (x0 === x1 && y0 === y1) break;
        const e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

// Map predifined gray values to pixels
function mapGrayValues(grayValues = []) {
    clear();
    
    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            const index = x * gridSize + y;
            if (index < grayValues.length) {
                pixels[y * gridSize + x].updateColor(grayValues[index]);
            }
        }
    }

    submit();
}

// Clear pixel grid
function clear() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
    draw_pixel_grid();
    result.innerHTML = '';

    for (let i = 0; i < probabilityBars.length; i++) {
        probabilityBars[i].style.width = '0%';
        probabilityBars[i].style.backgroundColor = 'rgba(0, 0, 0, 0)';
    }
}

// Map predicted probabilities to probability bars
function changeProbabilityBars(probabilities) {
    for (let i = 0; i < probabilities.length; i++) {
        probabilityBars[i].style.width = `${probabilities[i]}%`;
        probabilityBars[i].style.backgroundColor = `rgba(10, 100, 20, ${probabilities[i] / 100})`;
    }
}

// Get gray values of pixel grid and return as array
function getImageData() {
    const data = [];
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            data.push(pixels[j * gridSize + i].grayValue);
        }
    }
    return data;
}

// Submit image data to Network
function submit() {
    submit_button.click();
}

// Debounce submit function to prevent multiple requests
function debounceSubmit() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(submit, 300);
}

// Throttle submit function to prevent multiple requests
function throttleSubmit() {
    const now = new Date().getTime();
    const timeSinceLastCall = now - lastCall;

    if (timeSinceLastCall >= 500) {
        submit();
        lastCall = now;
    }
}


// Button Event listeners


submit_button.addEventListener('click', () => {
    const data = pixels.map(pixel => pixel.grayValue);
    // const data = [0, 1, 2]
    
    fetch('/clasify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_data: getImageData() })
    })
    .then(response => response.json())
    .then(data => {
        result.innerHTML = data.prediction;
        changeProbabilityBars(data.probabitlities)
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

random_button.addEventListener('click', () => {
    fetch('/loadRandom', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
    })
    .then(response => response.json())
    .then(data => {
        mapGrayValues(data.data)
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

clear_button.addEventListener('click', () => {
    clear();
});
