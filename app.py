import base64
import os

from flask import Flask, abort, flash, redirect, render_template, request, url_for

from ocr import radicals_db, recognize as rec

app = Flask(__name__)
# Only used to sign the flash() cookie (no auth/session data involved), but
# a random per-process key is still better hygiene than a fixed string now
# that the app is public. Set FLASK_SECRET_KEY to pin it across restarts.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize_view():
    photo = request.files.get("photo")
    if not photo or not photo.filename:
        flash("Файл не выбран.")
        return redirect(url_for("index"))

    image_bytes = photo.read()
    online_translate = request.form.get("online_translate") == "1"

    mimetype = photo.mimetype or "image/jpeg"
    photo_data_uri = f"data:{mimetype};base64," + base64.b64encode(image_bytes).decode("ascii")

    try:
        result = rec.recognize(image_bytes, use_online_translate=online_translate)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        result=result,
        photo_data_uri=photo_data_uri,
        online_translate=online_translate,
    )


@app.route("/char/<char>")
def char_view(char):
    char = char[:1]
    use_online_translate = request.args.get("translate") == "1"
    c = rec.resolve_char(char, use_online_translate=use_online_translate)
    return render_template("char_view.html", c=c)


@app.route("/browse")
def browse():
    q = request.args.get("q", "").strip()
    radicals = radicals_db.search(q)
    common_chars = radicals_db.search_common(q)
    return render_template("browse.html", radicals=radicals, common_chars=common_chars, q=q)


@app.route("/radical/<int:radical_id>")
def radical_detail(radical_id):
    r = radicals_db.get_by_id(radical_id)
    if not r:
        abort(404)
    return render_template("radical_detail.html", r=r)


@app.errorhandler(404)
def not_found(_exc):
    return render_template(
        "error.html",
        title="Страница не найдена",
        message="Такой страницы или ключа не существует.",
    ), 404


@app.errorhandler(413)
def too_large(_exc):
    mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return render_template(
        "error.html",
        title="Файл слишком большой",
        message=f"Максимальный размер фото — {mb} МБ. Сожми изображение и попробуй ещё раз.",
    ), 413


@app.errorhandler(500)
def server_error(_exc):
    return render_template(
        "error.html",
        title="Что-то пошло не так",
        message="Внутренняя ошибка сервера. Попробуй ещё раз чуть позже.",
    ), 500


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
