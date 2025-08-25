import os
import sys
import logging
import subprocess
import asyncio
from importlib import import_module
from datetime import datetime
from typing import List, Optional

# Step 1: شروع عملکرد ربات
if __name__ != '__main__':
    sys.exit(0)
# برای محیط‌هایی مانند Google Colab
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Step 2: بررسی وابستگی‌ها (پیش‌نصب توصیه می‌شود)
REQUIRED_LIBRARIES = {
    'python-telegram-bot': 'telegram',
    'instagrapi': 'instagrapi',
    'Pillow': 'PIL',
    'python-dotenv': 'dotenv',
    'moviepy': 'moviepy.editor',
    'filetype': 'filetype',
    'nest-asyncio': 'nest-asyncio',
    'sentry-sdk': 'sentry_sdk',
}

def check_and_warn_dependencies():
    missing = []
    for pkg, imp in REQUIRED_LIBRARIES.items():
        try:
            import_module(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        logging.warning(f"Missing dependencies: {', '.join(missing)}. Please install them before running the bot.")

check_and_warn_dependencies()

# Step 3: بارگذاری متغیرهای محیطی
from dotenv import load_dotenv
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
INSTAGRAM_USER = os.getenv('INSTAGRAM_USER')
INSTAGRAM_PASS = os.getenv('INSTAGRAM_PASS')
SENTRY_DSN = os.getenv('SENTRY_DSN')

if not all([TELEGRAM_TOKEN, INSTAGRAM_USER, INSTAGRAM_PASS]):
    missing = [n for n in ['TELEGRAM_TOKEN','INSTAGRAM_USER','INSTAGRAM_PASS'] if not os.getenv(n)]
    print(f"Missing .env variables: {', '.join(missing)}")
    sys.exit(1)

# تنظیم لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('bot.log'), logging.StreamHandler(sys.stdout)]
)
if SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(dsn=SENTRY_DSN)
    logging.info('Sentry initialized.')
else:
    logging.info('SENTRY_DSN not set, skipping Sentry.')

# Step 4: مدیریت پوشه دانلود
def prepare_downloads_folder() -> str:
    path = os.path.join(os.getcwd(), 'downloads')
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info('Downloads folder created.')
    else:
        for f in os.listdir(path):
            full_path = os.path.join(path, f)
            try:
                os.remove(full_path)
            except Exception as e:
                logging.debug(f'Failed to remove {full_path}: {e}')
        logging.info('Downloads folder cleared.')
    return path

downloads_path = prepare_downloads_folder()

# Step 5: مدیریت اتصال به اینستاگرام
from instagrapi import Client
from instagrapi.exceptions import TwoFactorRequired, ChallengeRequired

class InstagramManager:
    def __init__(self, username: str, password: str, session_file: str='ig_session.json'):
        self.client = Client()
        self.username = username
        self.password = password
        self.session_file = session_file
        self.attempts = 0

    def login(self, code_2fa: str=None, sms_code: str=None) -> tuple:
        if self.attempts >= 3:
            return False, 'ورود کد اشتباه بیش از 3 مرتبه'
        if self.load_session():
            self.attempts = 0
            return True, 'ورود با سشن موجود انجام شد.'
        try:
            args = {'username': self.username, 'password': self.password}
            if code_2fa:
                args['verification_code'] = code_2fa
            if sms_code:
                args['phone_number'] = sms_code
            if self.client.login(**args):
                self.save_session()
                self.attempts = 0
                return True, 'اتصال به اینستاگرام موفق بود!'
            return False, 'اتصال ناموفق'
        except TwoFactorRequired:
            self.attempts += 1
            return False, 'نیاز به کد 2FA'
        except ChallengeRequired as e:
            self.attempts += 1
            if 'sms' in str(e).lower():
                return False, 'نیاز به کد SMS'
            return False, f'خطای احراز هویت: {e}'
        except Exception as e:
            logging.exception('Instagram login error')
            return False, f'خطای سیستم: {e}'

    def load_session(self) -> bool:
        if os.path.exists(self.session_file):
            try:
                self.client.load_settings(self.session_file)
                self.client.get_timeline_feed()
                return True
            except Exception as e:
                logging.debug(f'Failed to load session: {e}')
                return False
        return False

    def save_session(self):
        self.client.dump_settings(self.session_file)

ig_manager = InstagramManager(INSTAGRAM_USER, INSTAGRAM_PASS)

# Step 6: تشخیص نوع فایل (نسخه سازگار با ویندوز)
class FileValidator:
    @staticmethod
    def validate(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()

        if ext == '.gif':
            return 'animation'

        if ext in ['.jpg','.jpeg','.png','.tiff','.bmp']:
            return 'image'

        if ext in ['.mp4','.avi','.flv','.webm','.mov','.mkv','.wmv']:
            return 'video'

        # بررسی محتوای فایل به عنوان مکانیزم ثانویه
        try:
            import filetype
            kind = filetype.guess(path)
            if kind:
                if kind.mime.startswith('image/'):
                    return 'image'
                if kind.mime.startswith('video/'):
                    return 'video'
        except ImportError:
            pass

        raise ValueError('فرمت فایل پشتیبانی نمی‌شود')

# Step 7: پردازش عکس
from PIL import Image
class ImageProcessor:
    @staticmethod
    def process(path: str) -> str:
        max_size = 1080
        with Image.open(path) as img:
            w, h = img.size
            scale = max_size / max(w, h)
            new_img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
            bg = Image.new('RGB', (max_size, max_size), (0, 0, 0))
            pos = ((max_size - new_img.width)//2, (max_size - new_img.height)//2)
            bg.paste(new_img, pos)
            out = f"{os.path.splitext(path)[0]}_processed.jpg"
            bg.save(out, format='JPEG', quality=100, optimize=True, subsampling=0)
            return out

# Step 8: پردازش ویدیو
import moviepy.editor as mp
class VideoProcessor:
    MAX_DURATION = 60
    @classmethod
    def process(cls, path: str) -> str:
        clip = mp.VideoFileClip(path)
        if clip.duration > cls.MAX_DURATION:
            clip.close()
            raise ValueError(f"زمان ویدئو بیش از {cls.MAX_DURATION} ثانیه است!")
        target = (1280,720) if clip.w >= clip.h else (720,1280)
        scale = min(target[0]/clip.w, target[1]/clip.h)
        resized = clip.resize(scale)
        bg = mp.ColorClip(size=target, color=(0,0,0), duration=clip.duration)
        final = mp.CompositeVideoClip([bg, resized.set_position('center')])
        out = f"{os.path.splitext(path)[0]}_processed.mp4"
        final.write_videofile(out, codec='libx264', preset='slow', ffmpeg_params=['-crf','18'], audio_codec='aac' if clip.audio else None, threads=8)
        clip.close(); final.close(); bg.close()
        return out

# Step 9: تبدیل GIF به MP4
class GIFConverter:
    @staticmethod
    def convert(path: str) -> str:
        clip = mp.VideoFileClip(path)
        out = f"{os.path.splitext(path)[0]}.mp4"
        clip.write_videofile(out, codec='libx264')
        clip.close()
        return out

# Step 10: تلگرام
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InputMediaPhoto, InputMediaVideo
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, ContextTypes

STATES = {'AUTH':0, 'AUTH_SMS':1, 'MEDIA_TYPE':2, 'RECEIVE_MEDIA':3, 'CONFIRM':4, 'CAPTION':5}

async def verify_telegram(app: Application) -> bool:
    try:
        await app.bot.get_me()
        return True
    except Exception as e:
        logging.error(f'Telegram auth failed: {e}')
        return False

# Cancel handler مشترک
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('♻️ عملیات لغو شد.', reply_markup=ReplyKeyboardRemove())
    prepare_downloads_folder()
    context.user_data.clear()
    return await send_welcome(update)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await verify_telegram(context.application):
        await update.message.reply_text('❌ خطا در اتصال به تلگرام!')
        return ConversationHandler.END
    success, msg = ig_manager.login()
    if success:
        await update.message.reply_text('✅ اتصال به اینستاگرام موفق بود!')
        return await send_welcome(update)
    if '2FA' in msg:
        await update.message.reply_text('🔐 لطفا کد 2FA را وارد کن:')
        return STATES['AUTH']
    if 'SMS' in msg:
        await update.message.reply_text('📱 لطفا کد SMS را وارد کن:')
        return STATES['AUTH_SMS']
    await update.message.reply_text(f'❌ {msg}')
    return ConversationHandler.END

async def handle_2fa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == '❌ لغو عملیات': return await cancel(update, context)
    ok, msg = ig_manager.login(code_2fa=update.message.text.strip())
    if ok:
        await update.message.reply_text('✅ احراز هویت موفق شد!')
        return await send_welcome(update)
    await update.message.reply_text(f'❌ {msg}')
    return STATES['AUTH']

async def handle_sms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == '❌ لغو عملیات': return await cancel(update, context)
    ok, msg = ig_manager.login(sms_code=update.message.text.strip())
    if ok:
        await update.message.reply_text('✅ احراز هویت موفق شد!')
        return await send_welcome(update)
    await update.message.reply_text(f'❌ {msg}')
    return STATES['AUTH_SMS']

async def send_welcome(update: Update) -> int:
    kb = [['📤 آلبوم','📎 تکی'], ['❌ لغو عملیات']]
    await update.message.reply_text('🤖 برای ارسال، حالت مورد نظر را انتخاب کن:', reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))
    prepare_downloads_folder()
    return STATES['MEDIA_TYPE']

async def handle_media_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == '❌ لغو عملیات': return await cancel(update, context)
    mode = 'album' if text.startswith('📤') else 'single'
    context.user_data['mode'] = mode

    # اصلاح پیام راهنما
    if mode == 'album':
        min_files = 2  # حداقل فایل برای آلبوم
        max_files = 10
        kb = [['🏁 اتمام','❌ لغو عملیات']]
        await update.message.reply_text(
            f'📤 لطفاً بین {min_files} تا {max_files} فایل ارسال کن:',
            reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))
    else:
        min_files = max_files = 1
        kb = [['❌ لغو عملیات']]
        await update.message.reply_text(
            f'📎 لطفاً یک فایل ارسال کن:',
            reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))

    return STATES['RECEIVE_MEDIA']

media_counter = 1

async def download_media(update: Update) -> Optional[str]:
    global media_counter
    msg = update.message
    if msg.photo:
        media, ext = msg.photo[-1], '.jpg'
    elif msg.video:
        media, ext = msg.video, '.mp4'
    elif msg.animation:
        media, ext = msg.animation, '.gif'
    else:
        await msg.reply_text('⚠️ فرمت نامعتبر!')
        return None

    file = await media.get_file()
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{media_counter:03d}{ext}"
    media_counter += 1

    path = os.path.join(downloads_path, name)
    await file.download_to_drive(path)
    logging.info(f'Downloaded: {path}')
    return path

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == 'لغو عملیات': return await cancel(update, context)
    if text == '🏁 اتمام': return await process_media(update, context)
    files = context.user_data.setdefault('files', [])
    if context.user_data['mode']=='single' and files:
        await update.message.reply_text('⚠️ فقط یک فایل مجاز است!')
        return STATES['RECEIVE_MEDIA']
    path = await download_media(update)
    if not path: return STATES['RECEIVE_MEDIA']
    files.append(path)
    if len(files) >= (10 if context.user_data['mode']=='album' else 1):
        return await process_media(update, context)
    rem = 10 - len(files) if context.user_data['mode']=='album' else 0
    await update.message.reply_text(f'✅ دریافت شد. باقی: {rem}')
    return STATES['RECEIVE_MEDIA']

async def process_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('⏳ در حال پردازش...')
    processed = []
    for p in context.user_data.get('files', []):
        try:
            t = FileValidator.validate(p)
            if t=='animation':
                p = await asyncio.to_thread(GIFConverter.convert, p)
                t = 'video'
            out = await asyncio.to_thread((ImageProcessor.process if t=='image' else VideoProcessor.process), p)
            processed.append(out)
        except Exception as e:
            await update.message.reply_text(f'❌ خطا: {e}')
    if not processed:
        return await send_welcome(update)
    context.user_data['processed'] = processed
    return await send_previews(update, processed)

async def send_previews(update: Update, files: List[str]) -> int:
    media = []
    for f in files:
        if f.endswith('.jpg'):
            media.append(InputMediaPhoto(open(f,'rb')))
        else:
            media.append(InputMediaVideo(open(f,'rb')))
    await update.message.reply_media_group(media)
    kb = [['✅ تایید','❌ لغو عملیات']]
    await update.message.reply_text('آیا ادامه دهیم؟', reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))
    return STATES['CONFIRM']

async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == '❌ لغو عملیات': return await cancel(update, context)

    if text == '✅ تایید':
        files = context.user_data.get('processed', [])
        mode = context.user_data.get('mode')

        # هشدار به کاربر اگر آلبوم کمتر از 2 فایل داشته باشد
        if mode == 'album' and len(files) < 2:
            await update.message.reply_text('⚠️ برای آپلود آلبوم حداقل ۲ فایل نیاز است!')
            return await send_welcome(update)

        await update.message.reply_text('📝 لطفاً کپشن را وارد کن:', reply_markup=ReplyKeyboardRemove())
        return STATES['CAPTION']

    return STATES['CONFIRM']

async def handle_caption(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if text == '❌ لغو عملیات': return await cancel(update, context)
    cap = text
    try:
        files = context.user_data.get('processed', [])
        mode = context.user_data.get('mode')

        if mode == 'album':
            # بررسی تعداد فایل‌ها برای آلبوم
            if len(files) < 2:
                await update.message.reply_text('⚠️ برای آپلود آلبوم حداقل ۲ فایل نیاز است!')
                return await send_welcome(update)

            ig_manager.client.album_upload(files, caption=cap)
        else:
            # آپلود تکی
            for f in files:
                if f.endswith('.jpg'):
                    ig_manager.client.photo_upload(f, caption=cap)
                else:
                    ig_manager.client.video_upload(f, caption=cap)

        await update.message.reply_text('✅ آپلود موفقیت‌آمیز بود!')
    except Exception as e:
        logging.exception('Upload error')
        await update.message.reply_text(f'❌ خطا در آپلود: {e}')
    return await send_welcome(update)

# اجرای ربات
app = Application.builder().token(TELEGRAM_TOKEN).build()
conv_handler = ConversationHandler(
    entry_points=[CommandHandler('start', start)],
    states={
        STATES['AUTH']:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_2fa)],
        STATES['AUTH_SMS']:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_sms)],
        STATES['MEDIA_TYPE']:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_media_type)],
        STATES['RECEIVE_MEDIA']:[
            MessageHandler(filters.PHOTO|filters.VIDEO|filters.ANIMATION, handle_media),
            MessageHandler(filters.TEXT & filters.Regex(r'^🏁 اتمام$'), process_media),
            MessageHandler(filters.Regex('^❌ لغو عملیات$'), cancel)
        ],
        STATES['CONFIRM']:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_confirmation)],
        STATES['CAPTION']:[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_caption)]
    },
    fallbacks=[CommandHandler('cancel', cancel)],
    conversation_timeout=24*60,
    allow_reentry=True
)
app.add_handler(conv_handler)
app.run_polling()