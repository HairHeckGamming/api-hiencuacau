const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    // 1. ĐỊNH DANH CỐ ĐỊNH
    username: { type: String, required: true, unique: true, lowercase: true, trim: true },
    
    // 2. TÊN HIỂN THỊ
    displayName: { type: String, trim: true },

    // 3. THÔNG TIN CƠ BẢN
    email: { type: String, required: true, trim: true },
    password: { type: String, required: true },
    avatar: { type: String, default: "" },

    // 4. NHẬN DIỆN GOOGLE OAUTH
    hwid: { 
        type: String, 
        unique: true, 
        sparse: true, // ⚡ CHÌA KHÓA GIẢI QUYẾT LỖI Ở ĐÂY
        default: null 
    },

    // 5. BẢO MẬT & QUÊN MẬT KHẨU (OTP)
    resetPasswordOtp: { type: String, default: null },
    resetPasswordExpires: { type: Date, default: null },

    // 6. HỒ SƠ TÂM LÝ & TRÍ NHỚ DÀI HẠN (CỐT LÕI CỦA AI)
    userContext: { 
        type: String, 
        default: "Người dùng mới, chưa có thông tin bối cảnh cụ thể." 
    },
    // 6. HỒ SƠ TÂM LÝ & TRÍ NHỚ DÀI HẠN (CỐT LÕI CỦA AI)
    userContext: { 
        type: String, 
        default: "Người dùng mới, chưa có thông tin bối cảnh cụ thể." 
    },
    // ---> THÊM TRƯỜNG NÀY: QUY ĐỊNH RANH GIỚI TRỊ LIỆU <---
    aiPersona: {
        type: String,
        enum: ['hugging', 'socratic', 'tough_love'],
        default: 'hugging'
    },
    isIncognito: {
        type: Boolean,
        default: false
    },

    aiMemory: {
        // 1. Warm Continuity (Kết nối cảm xúc)
        warmContinuity: { type: Array, default: [] }, // Format: { content: String, emotion: String, date: String }
        
        // 2. Safety Flag (Bảo vệ tính mạng ngầm)
        hasPastHighRisk: { type: Boolean, default: false },
        
        // 3. Comfort Preference (Sở thích được xoa dịu)
        bestApproach: { type: String, default: "hugging" }, // hugging, socratic, silence, micro
        
        // 4. Trigger Patterns (Điểm kích hoạt cảm xúc)
        triggerPatterns: { type: Array, default: [] } // Format: { trigger: "Học tập", emotion: "anxiety" }
    },
    
    // ĐÂY LÀ VÙNG TRÍ NHỚ MỚI CỦA CẤP ĐỘ 1:
    // Nơi AI tự động đúc kết và nhét các sự kiện quan trọng vào.
    coreMemories: {
        type: [String],
        default: []
    },

    blacklistedTopics: {
        type: [String],
        default: []
    },

    // 7. DỮ LIỆU CÁC CÔNG CỤ TRỊ LIỆU
    moodHistory: { type: Array, default: [] },
    fireflies: { type: Array, default: [] },
    microWinsCount: { type: Number, default: 0 },
    
    // THÊM 2 DÒNG NÀY CHO CÂY THẾ GIỚI
    totalEnergy: { type: Number, default: 0 },
    rebirthCount: { type: Number, default: 0 },
    
    // THÊM DÒNG NÀY: Nhật ký lưu năng lượng theo từng ngày
    energyHistory: { type: Array, default: [] },
}, { timestamps: true });

module.exports = mongoose.model('User', userSchema);