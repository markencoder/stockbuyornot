# Deployment

This project is a Streamlit web app. For the first commercial MVP, deploy it as a single web service with persistent storage.

## Recommended for mainland China: Tencent Cloud Lighthouse

For users in mainland China, use a domestic lightweight server first. Tencent Cloud Lighthouse or Alibaba Cloud Simple Application Server both work. This repository includes `Dockerfile` and `docker-compose.yml`, so the deployment flow is the same on either platform.

### Server

Recommended first MVP size:

- Ubuntu 22.04 LTS
- 2 vCPU / 4 GB RAM
- 40 GB SSD or larger
- Open firewall port `8501` for testing

### Deploy with Docker Compose

```bash
sudo apt update
sudo apt install -y ca-certificates curl git docker.io docker-compose-plugin
sudo systemctl enable --now docker

git clone https://github.com/YOUR_NAME/stockbuyornot.git
cd stockbuyornot

sudo docker compose up -d --build
sudo docker compose logs -f
```

Then open:

```text
http://YOUR_SERVER_IP:8501
```

### Configure Payment

Edit `docker-compose.yml`:

```yaml
STOCKBUYORNOT_PAYMENT_QR_URL: "https://your-domain.com/payment-qr.png"
STOCKBUYORNOT_SUPPORT_CONTACT: "your-wechat-or-email"
STOCKBUYORNOT_BILLING_ENFORCED: "true"
```

Then restart:

```bash
sudo docker compose up -d
```

The `./data` directory is mounted into the container, so `app.db` and each user's files persist across restarts.

### Domain and ICP

If you only use `http://server-ip:8501`, you can test immediately. For a public custom domain in mainland China, complete ICP filing before long-term production use.

After ICP and DNS are ready, add Nginx and HTTPS in front of Streamlit.

## Optional: Render outside mainland China

1. Push this repository to GitHub.
2. Create a Render account and choose **New +** > **Blueprint**.
3. Select this repository. Render will read `render.yaml`.
4. After the service is created, open **Environment** and set:
   - `STOCKBUYORNOT_PAYMENT_QR_URL`: your payment QR image URL
   - `STOCKBUYORNOT_SUPPORT_CONTACT`: your WeChat, email, or support contact
   - `STOCKBUYORNOT_BILLING_ENFORCED`: set to `true` when you want paid features locked
5. Deploy the service.
6. Add your custom domain in Render, then configure the DNS record at your domain provider.

The persistent disk is mounted at `/opt/render/project/src/data`, so the SQLite user database and each user's watchlist files survive restarts and deploys.

## Start command

```bash
streamlit run src/stockbuyornot/ui/streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --browser.gatherUsageStats=false
```

## Production Notes

- SQLite is acceptable for the first small MVP. Move to PostgreSQL before heavy traffic or multi-instance scaling.
- Keep `STOCKBUYORNOT_BILLING_ENFORCED=false` while testing. Turn it on after your payment flow and support process are ready.
- Do not commit `data/app.db` or `data/users/`; they contain user data.
- Keep the investment risk disclaimer visible in the product and user agreement.
