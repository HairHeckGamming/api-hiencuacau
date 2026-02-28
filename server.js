require('dotenv').config(); // Tải các biến môi trường từ file .env
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

// Khởi tạo ứng dụng Express
const app = express();

const memoryRoutes = require('./routes/memoryRoutes');

// ==========================================
// 1. CẤU HÌNH MIDDLEWARE & CORS
// ==========================================
// ⚡ NỚI LỎNG GIỚI HẠN LÊN 50MB ĐỂ CHỨA VỪA ẢNH BASE64
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Cấu hình CORS để Frontend (Render) và Backend có thể nói chuyện với nhau
app.use(cors({
    origin: [
        'https://hiencuacau.onrender.com', // Link Frontend thật trên Render
        'http://localhost:5173',           // Link Local của Vite để cậu test trên máy
        'http://localhost:3000'            // Link Local dự phòng
    ],
    credentials: true
}));

// ==========================================
// 2. KẾT NỐI DATABASE (MONGODB)
// ==========================================
mongoose.connect(process.env.MONGODB_URI)
    .then(async () => {
        console.log('🌿 Đã kết nối thành công với kho lưu trữ Hiên Của Cậu (MongoDB)!');
        
        // 🚀 TIÊM THUỐC GIẢI: Ra lệnh xóa cái index cũ đang gây lỗi
        try {
            await mongoose.connection.collection('sessions').dropIndex('sessionId_1');
            console.log('✨ Đã dọn dẹp thành công tàn dư sessionId_1 cũ!');
        } catch (e) {
            // Nếu nó báo lỗi thì tức là index đã được xóa rồi, không sao cả
        }
    })
    .catch((err) => console.error('🚨 Lỗi kết nối MongoDB:', err));

// ==========================================
// 3. ĐƯỜNG DÂY NÓNG GIỮ SERVER LUÔN THỨC (PINGER)
// ==========================================
// Route này dùng để UptimeRobot hoặc Frontend gọi vào để giữ server không bị ngủ đông
app.get('/api/ping', (req, res) => {
    res.status(200).json({ 
        status: "ready", 
        message: "Hiên Của Cậu đã sẵn sàng đón khách! 🌿",
        timestamp: new Date()
    });
});

// ==========================================
// 4. ĐIỀU PHỐI ĐƯỜNG DẪN (ROUTES)
// ==========================================
// Chuyển hướng các yêu cầu Đăng ký/Đăng nhập/Google sang file authRoutes.js
app.use('/api/auth', require('./routes/authRoutes'));

// Chuyển hướng các yêu cầu Trò chuyện với AI sang file chatRoutes.js
app.use('/api/chat', require('./routes/chatRoutes'));

const userRoutes = require('./routes/userRoutes');
app.use('/api/user', userRoutes);

// Thêm các phòng ban khác nếu cậu có làm (Ví dụ: Nhật ký, Lọ đom đóm...)
// app.use('/api/user', require('./routes/userRoutes')); 

app.use('/api', require('./routes/toolRoutes'));

app.use('/api/memories', memoryRoutes);

// ==========================================
// 5. BẮT LỖI TOÀN CỤC (GLOBAL ERROR HANDLER)
// ==========================================
// Nếu người dùng gọi vào một đường link không tồn tại
app.use((req, res, next) => {
    res.status(404).json({ message: "Đường dẫn này không tồn tại trong Hiên Của Cậu." });
});

// Bắt các lỗi sập ngầm của Server
app.use((err, req, res, next) => {
    console.error("🚨 Lỗi Server Nghiêm Trọng:", err.stack);
    res.status(500).json({ message: "Có lỗi xảy ra ở hệ thống trung tâm. Cậu đợi một lát nhé." });
});

// ==========================================
// 6. KHỞI ĐỘNG SERVER
// ==========================================
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`🚀 Hệ thống đang chạy tại Port ${PORT}`);
    console.log(`👉 Bấm vào đây để test Ping: http://localhost:${PORT}/api/ping`);
});