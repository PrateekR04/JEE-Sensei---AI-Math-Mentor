# 🚀 Deployment Guide for JEE Sensei

## Step 1: Push to GitHub

### Initialize Git (if not already done)
```bash
cd /Users/prateekroshan/Desktop/AI\ Planet\ Assessment/math_mentor_ai
git init
```

### Add all files
```bash
git add .
git commit -m "Initial commit: JEE Sensei - AI Math Assistant"
```

### Create GitHub Repository
1. Go to [github.com/new](https://github.com/new)
2. Create a new repository named `jee-sensei` (or any name you prefer)
3. **DO NOT** initialize with README (we already have one)

### Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/jee-sensei.git
git branch -M main
git push -u origin main
```

---

## Step 2: Deploy to Streamlit Cloud

### 2.1 Go to Streamlit Cloud
- Visit [share.streamlit.io](https://share.streamlit.io)
- Sign in with your GitHub account

### 2.2 Create New App
1. Click **"New app"**
2. Select your repository: `YOUR_USERNAME/jee-sensei`
3. Branch: `main`
4. Main file path: `app.py`
5. Click **"Deploy!"**

### 2.3 Add Secrets (IMPORTANT!)
Before the app runs, you need to add your API key:

1. In Streamlit Cloud dashboard, click on your app
2. Click **"Settings"** ⚙️
3. Click **"Secrets"** in the left sidebar
4. Add the following:

```toml
GROQ_API_KEY = "your_actual_groq_api_key_here"
```

5. Click **"Save"**

### 2.4 Restart App
After adding secrets, the app will automatically restart with your API key configured.

---

## 📋 Pre-Deployment Checklist

- [ ] `.env` file is in `.gitignore` (API keys not exposed)
- [ ] `requirements.txt` includes all dependencies
- [ ] `.streamlit/config.toml` exists for theme settings
- [ ] No hardcoded API keys in code
- [ ] Test app locally before deploying

---

## 🔒 Security Notes

1. **Never commit `.env` file** - It contains your API keys
2. **Use Streamlit Secrets** - Add API keys in Streamlit Cloud dashboard
3. **Rotate keys if exposed** - If you accidentally push a key, revoke it immediately

---

## 🔗 Getting a Groq API Key

If you need a Groq API key:
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up / Sign in
3. Navigate to **API Keys**
4. Create a new API key
5. Copy and save it securely

---

## 📝 Troubleshooting

### "No module named 'X'" Error
- Make sure all dependencies are in `requirements.txt`
- Restart the app in Streamlit Cloud

### API Key Not Working
- Check Secrets are correctly configured
- Verify the key format (should start with `gsk_` for Groq)

### Memory Issues
- Streamlit Cloud has memory limits
- Large model downloads may timeout on first run
