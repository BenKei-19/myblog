# Stage 1: Development with Jekyll 4 official base image (đầy đủ Ruby & build tools)
FROM jekyll/jekyll:4 AS dev

WORKDIR /srv/jekyll

# Copy dependency files
COPY Gemfile Gemfile.lock* ./

# Cài đặt các plugin / gem được khai báo trong Gemfile
RUN bundle install

# Copy toàn bộ mã nguồn blog
COPY . .

# Mở cổng Jekyll và LiveReload
EXPOSE 4000 35729

# Chạy server ở root URL (baseurl="") để truy cập localhost:4000 không bị 404
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--port", "4000", "--baseurl", "", "--livereload", "--force_polling"]

# Stage 2: Build static site cho Production với baseurl rỗng
FROM dev AS builder
ENV JEKYLL_ENV=production
RUN bundle exec jekyll build --baseurl ""

# Stage 3: Web server Nginx siêu nhẹ (~25MB) phục vụ Production
FROM nginx:alpine AS prod
# Copy cấu hình Nginx tối ưu routing
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /srv/jekyll/_site /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
