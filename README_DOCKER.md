# Hướng dẫn chạy và Host Blog qua Docker

Dự án Jekyll Blog của bạn đã được cấu hình sẵn môi trường Docker hoàn chỉnh, không cần cài thêm Ruby, Gem hay bất cứ dependency nào trên máy thật.

---

## 🚀 1. Chế độ Phát triển / Viết bài (Development với Hot-Reload)

Dùng khi bạn đang viết bài hoặc chỉnh sửa giao diện và muốn xem thay đổi tức thì ngay khi lưu file.

### Khởi chạy:
```bash
docker compose up myblog-dev
```
*(Nếu muốn chạy ngầm dưới nền, thêm cờ `-d`: `docker compose up -d myblog-dev`)*

### Truy cập:
- **Địa chỉ:** [http://localhost:4000](http://localhost:4000) (hoặc [http://localhost:4000/myblog/](http://localhost:4000/myblog/))
- **LiveReload:** Tự động reload trình duyệt khi bạn thay đổi các file trong `_posts`, `_layouts`, `assets`, `index.markdown`...

---

## 🌐 2. Chế độ Triển khai / Host thực tế (Production với Nginx)

Dùng khi bạn muốn host blog ổn định, tốc độ load siêu nhanh và bảo mật thông qua Web Server **Nginx** (dung lượng cực nhẹ ~25MB).

### Khởi chạy:
```bash
docker compose up -d myblog-prod
```

### Truy cập:
- **Địa chỉ:** [http://localhost:8080](http://localhost:8080)

---

## 🛠️ 3. Các lệnh hữu ích thường dùng

| Thao tác | Câu lệnh |
| :--- | :--- |
| **Dừng tất cả container** | `docker compose down` |
| **Xem log trực tiếp (Dev)** | `docker logs -f myblog_dev` |
| **Xem log trực tiếp (Prod)** | `docker logs -f myblog_prod` |
| **Build lại sau khi thêm Gem mới** | `docker compose build --no-cache` |
| **Kiểm tra trạng thái container** | `docker ps` |
