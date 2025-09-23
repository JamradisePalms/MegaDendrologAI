// Global variables
let selectedFiles = [];
let uploadedPhotos = JSON.parse(localStorage.getItem('uploadedPhotos')) || [];
let consultantOpen = false;

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeUpload();
    loadGallery();
    initializeConsultant();
});

// Navigation functionality
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = e.target.dataset.page;
            showPage(page);
            
            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            e.target.classList.add('active');
        });
    });
}

function showPage(pageName) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => page.classList.remove('active'));
    
    const targetPage = document.getElementById(`${pageName}-page`);
    if (targetPage) {
        targetPage.classList.add('active');
    }
}

function showUploadPage() {
    showPage('upload');
    document.querySelector('[data-page="upload"]').classList.add('active');
    document.querySelector('[data-page="home"]').classList.remove('active');
}

// Upload functionality
function initializeUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    
    // Click to select files
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

function handleFiles(files) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    Array.from(files).forEach(file => {
        if (!validTypes.includes(file.type)) {
            alert(`Файл ${file.name} имеет неподдерживаемый формат`);
            return;
        }
        
        if (file.size > maxSize) {
            alert(`Файл ${file.name} слишком большой (максимум 10MB)`);
            return;
        }
        
        selectedFiles.push(file);
    });
    
    updatePreview();
}

function updatePreview() {
    const previewSection = document.getElementById('preview-section');
    const previewGrid = document.getElementById('preview-grid');
    
    if (selectedFiles.length === 0) {
        previewSection.style.display = 'none';
        return;
    }
    
    previewSection.style.display = 'block';
    previewGrid.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            previewItem.innerHTML = `
                <img src="${e.target.result}" alt="Preview">
                <button class="remove-btn" onclick="removeFile(${index})">
                    <i class="fas fa-times"></i>
                </button>
            `;
            previewGrid.appendChild(previewItem);
        };
        reader.readAsDataURL(file);
    });
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updatePreview();
}

function uploadPhoto() {
    if (selectedFiles.length === 0) {
        alert('Пожалуйста, выберите файлы для загрузки');
        return;
    }
    
    const title = document.getElementById('photo-title').value || 'Без названия';
    const description = document.getElementById('photo-description').value || '';
    const location = document.getElementById('photo-location').value || '';
    
    selectedFiles.forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const photo = {
                id: Date.now() + Math.random(),
                title: title,
                description: description,
                location: location,
                image: e.target.result,
                uploadDate: new Date().toLocaleDateString('ru-RU')
            };
            
            uploadedPhotos.push(photo);
            localStorage.setItem('uploadedPhotos', JSON.stringify(uploadedPhotos));
        };
        reader.readAsDataURL(file);
    });
    
    // Clear form
    selectedFiles = [];
    document.getElementById('photo-title').value = '';
    document.getElementById('photo-description').value = '';
    document.getElementById('photo-location').value = '';
    updatePreview();
    
    alert('Фотографии успешно загружены!');
    setTimeout(() => {
        showPage('gallery');
        document.querySelector('[data-page="gallery"]').classList.add('active');
        document.querySelector('[data-page="upload"]').classList.remove('active');
        loadGallery();
    }, 1000);
}

// Gallery functionality
function loadGallery() {
    const galleryGrid = document.getElementById('gallery-grid');
    
    // Sample photos if no uploaded photos exist
    const samplePhotos = [
        {
            id: 1,
            title: 'Могучий дуб',
            description: 'Столетний дуб в центральном парке города',
            location: 'Москва, Центральный парк',
            image: 'https://images.pexels.com/photos/268533/pexels-photo-268533.jpeg?auto=compress&cs=tinysrgb&w=600',
            uploadDate: '15.01.2025'
        },
        {
            id: 2,
            title: 'Березовая роща',
            description: 'Красивая березовая роща весной',
            location: 'Подмосковье',
            image: 'https://images.pexels.com/photos/414171/pexels-photo-414171.jpeg?auto=compress&cs=tinysrgb&w=600',
            uploadDate: '14.01.2025'
        },
        {
            id: 3,
            title: 'Сосновый лес',
            description: 'Величественные сосны в утреннем свете',
            location: 'Карелия',
            image: 'https://images.pexels.com/photos/1496372/pexels-photo-1496372.jpeg?auto=compress&cs=tinysrgb&w=600',
            uploadDate: '13.01.2025'
        },
        {
            id: 4,
            title: 'Осенний клен',
            description: 'Яркие осенние краски клена',
            location: 'Санкт-Петербург',
            image: 'https://images.pexels.com/photos/33109/fall-autumn-red-season.jpg?auto=compress&cs=tinysrgb&w=600',
            uploadDate: '12.01.2025'
        }
    ];
    
    const photosToShow = uploadedPhotos.length > 0 ? uploadedPhotos : samplePhotos;
    
    galleryGrid.innerHTML = '';
    
    photosToShow.forEach(photo => {
        const galleryItem = document.createElement('div');
        galleryItem.className = 'gallery-item';
        galleryItem.innerHTML = `
            <img src="${photo.image}" alt="${photo.title}">
            <div class="gallery-item-info">
                <h4>${photo.title}</h4>
                <p>${photo.description}</p>
                <p class="location"><i class="fas fa-map-marker-alt"></i> ${photo.location}</p>
                <p style="font-size: 0.8rem; color: #999; margin-top: 0.5rem;">
                    Загружено: ${photo.uploadDate}
                </p>
            </div>
        `;
        
        galleryItem.addEventListener('click', () => openPhotoModal(photo));
        galleryGrid.appendChild(galleryItem);
    });
}

function openPhotoModal(photo) {
    // Create modal for photo viewing
    const modal = document.createElement('div');
    modal.className = 'photo-modal';
    modal.innerHTML = `
        <div class="modal-overlay" onclick="closePhotoModal()">
            <div class="modal-content" onclick="event.stopPropagation()">
                <button class="modal-close" onclick="closePhotoModal()">
                    <i class="fas fa-times"></i>
                </button>
                <img src="${photo.image}" alt="${photo.title}">
                <div class="modal-info">
                    <h3>${photo.title}</h3>
                    <p>${photo.description}</p>
                    <p class="location"><i class="fas fa-map-marker-alt"></i> ${photo.location}</p>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Add modal styles
    const modalStyles = `
        <style>
            .photo-modal {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                z-index: 2000;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .modal-overlay {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.8);
            }
            .modal-content {
                background: white;
                border-radius: 20px;
                max-width: 90vw;
                max-height: 90vh;
                overflow: hidden;
                position: relative;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            }
            .modal-close {
                position: absolute;
                top: 1rem;
                right: 1rem;
                background: rgba(255, 255, 255, 0.9);
                border: none;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                cursor: pointer;
                z-index: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.2rem;
            }
            .modal-content img {
                width: 100%;
                max-height: 60vh;
                object-fit: contain;
            }
            .modal-info {
                padding: 1.5rem;
            }
            .modal-info h3 {
                color: var(--primary-green);
                margin-bottom: 0.5rem;
            }
            .modal-info .location {
                color: var(--light-green);
                font-weight: 600;
                margin-top: 0.5rem;
            }
        </style>
    `;
    
    document.head.insertAdjacentHTML('beforeend', modalStyles);
}

function closePhotoModal() {
    const modal = document.querySelector('.photo-modal');
    if (modal) {
        modal.remove();
    }
}

// Consultant functionality
function initializeConsultant() {
    const messageInput = document.getElementById('message-input');
    
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Auto-responses
    setTimeout(() => {
        if (!consultantOpen) {
            showNotification();
        }
    }, 30000);
}

function toggleConsultant() {
    const consultantChat = document.getElementById('consultant-chat');
    consultantOpen = !consultantOpen;
    
    if (consultantOpen) {
        consultantChat.classList.add('active');
    } else {
        consultantChat.classList.remove('active');
    }
}

function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    const chatMessages = document.getElementById('chat-messages');
    const now = new Date();
    const timeString = now.getHours().toString().padStart(2, '0') + ':' + 
                      now.getMinutes().toString().padStart(2, '0');
    
    // Add user message
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.innerHTML = `
        <div class="message-content">${message}</div>
        <div class="message-time">${timeString}</div>
    `;
    chatMessages.appendChild(userMessage);
    
    messageInput.value = '';
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Generate response
    setTimeout(() => {
        const response = generateResponse(message);
        const consultantMessage = document.createElement('div');
        consultantMessage.className = 'message consultant-message';
        consultantMessage.innerHTML = `
            <div class="message-content">${response}</div>
            <div class="message-time">${timeString}</div>
        `;
        chatMessages.appendChild(consultantMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 1000);
}

function generateResponse(message) {
    const responses = {
        'привет': 'Привет! Рад вас видеть. Чем могу помочь в мире деревьев?',
        'дерево': 'О деревьях можно говорить бесконечно! Что именно вас интересует?',
        'как': 'Отличный вопрос! Я помогу вам разобраться.',
        'спасибо': 'Пожалуйста! Всегда рад помочь с вопросами о природе.',
        'фото': 'Загружайте свои фотографии деревьев в разделе "Загрузить фото"!',
        'дуб': 'Дубы - это величественные деревья, символ силы и долголетия. Могут жить более 500 лет!',
        'береза': 'Березы - символ России! Эти красивые белоствольные деревья очень полезны для экосистемы.',
        'сосна': 'Сосны - вечнозеленые хвойные деревья. Отличный источник чистого воздуха!',
        'лес': 'Лес - это целая экосистема! Деревья, растения, животные - все взаимосвязано.'
    };
    
    const lowerMessage = message.toLowerCase();
    
    for (const [key, response] of Object.entries(responses)) {
        if (lowerMessage.includes(key)) {
            return response;
        }
    }
    
    return 'Интересный вопрос! Расскажите подробнее, и я постараюсь помочь вам с информацией о деревьях и лесе.';
}

function showNotification() {
    const consultantToggle = document.querySelector('.consultant-toggle');
    consultantToggle.style.animation = 'pulse 1s infinite';
    
    setTimeout(() => {
        consultantToggle.style.animation = '';
    }, 5000);
}

// Utility functions
function formatDate(date) {
    return date.toLocaleDateString('ru-RU', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
    });
}

// Add pulse animation
const pulseAnimation = `
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = pulseAnimation;
document.head.appendChild(styleSheet);