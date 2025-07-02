import requests


def download_yadisk(public_url: str, save_path: str):
    """Функция для загрузки файлов с Яндекс диска по публичной ссылке"""

    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {'public_key': public_url}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        download_url = response.json()['href']
        print("Скачиваем с:", download_url)
        r = requests.get(download_url)
        with open(save_path, 'wb') as f:
            f.write(r.content)
        return save_path
    else:
        return None