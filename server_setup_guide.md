# Виртуалка
1. Устанавливаем multipass
2. Создаем инстанс 18.04 `multipass launch -n instance_name -d 12G -m 1536M bionic`
## Reference:
* [Multipass](https://multipass.run/)

# Настраиваем SSH подключение
* Для аутентификации используем RSA-ключи
* Ключи хранятся в `~/.ssh`; `id_rsa` - приватный (для дешифровки), `id_rsa.pub` - публичный, который служит для проверки идентичности пользовател
1. Заходим в инстанс `multipass shell instance_name`
2. Копируем публичный ключ клиента на сервер `echo public_key >> ~/.ssh/authorized_keys`
3. Запускаем фаерволл и разрешаем подключение по SSH
 * `sudo ufw allow OpenSSH`
 * `sudo ufw enable`
4. Теперь можно подключаться с клиента по SSH `ssh ubuntu@ip`
## Reference:
* [SSH on Ubuntu](https://help.ubuntu.ru/wiki/ssh)
* [Как добавить ключ на сервер](https://firstvds.ru/technology/dobavit-ssh-klyuch)

# Настраиваем виртуальную среду
* В качестве дистрибутива используем miniconda3 (для использования менеджера разрешения зависимостей conda); py37
1. Скачиваем дистрибутив `wget -P ~/Downloads https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh`
2. Устанавливаем дистрибутив `bash ~/Downloads/Miniconda3-py37_4.9.2-Linux-x86_64.sh` в стандартную директорию. Не инициализируем
3. Создаем папку для проекта `mkdir ~/project`
4. Переносим с клиента на сервер requirements `scp pr/tab_net_mmdet/docker/requirements.txt ubuntu@ip:~/project/`
5. Заходим в окружение conda `. miniconda3/bin/activate`
6. Добавляем репозиторий библиотек `conda config --add channels conda-forge`
7. Устанавливаем виртуальное окружение для проекта `conda create -n project_env --file ~/project/requirements.txt`
8. Перемещаем проект в `~/project`. Главное приложение - `~/project/dash_app.py`. Указываем внутри `server = app.server` (инстанс Flask). Для теста прописываем `app.run_server(host='0.0.0.0')` и открываем порт. Приложение будет по адресу сервера.
## Reference:
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* [Miniconda repo](https://repo.anaconda.com/miniconda/)
* [Dash test app](https://dash.plotly.com/layout)

# Настраиваем WSGI (Gunicorn)
1. Создаем точку входа WSGI `nano ~/project/wsgi.py`. Заполняем как указано в reference. Для dash вместо `app` указываем `server`. Для проверки Gunicorn смотрим `gunicorn --bind 0.0.0.0:8050 wsgi:server`.
2. Выходим из виртуальной среды `conda deactivate`
3. Создаем сервис как указано в reference
 * Conda: `Environment="PATH=/home/ubuntu/miniconda3/envs/project_env/bin"`
 * Conda: `ExecStart=/home/ubuntu/miniconda3/envs/project_env/bin/gunicorn --workers 3 --bind unix:project.sock -m 007 wsgi:server`
4. Запускаем службу Gunicorn:
 * `sudo systemctl start project`
 * `sudo systemctl enable project`
 * `sudo systemctl status project`
## Reference:
* [Статья на DigitalOcean для стека Flask+Gunicorn+Nginx](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04-ru)
* [Additional: Deploying Python+Flask](http://wiki.ieo.es/books/centro-nacional-de-datos-oceanogr%C3%A1ficos/page/deploying-python-flask-the-example-of-ieo-ctd-checker)

# Настраиваем Nginx
1. Устанавливаем Nginx `sudo apt update && sudo apt install nginx`
2. Открываем порт 80 `sudo ufw allow 'Nginx HTTP'`
3. Меняем порт с 80 на 81 у дефолтной конфигурации `etc/nginx/sites-available/default`
4. Настраиваем Nginx как указано в reference (project вместо myproject).
## Reference:
* [Install Nginx on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04)
* [Статья на DigitalOcean для стека Flask+Gunicorn+Nginx](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04-ru)

# Выводим localhost в паблик с помощью Ngrok
1. Устанавливаем Ngrok:
 * `wget -P ~/Downloads https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip`
 * `sudo apt install unzip && unzip ~/Downloads/ngrok-stable-linux-amd64.zip`
 * `sudo mv ngrok /usr/local/bin`
2. Коннектим аккаунт с помощью аутентификационного ключа `./ngrok authtoken <your_auth_token>`
3. Добавляем порт аналитического интерфейса в фаерволл `sudo ufw allow 4040`
4. Создаем конфиг для доступа к аналитическому веб интерфейсу сервера на виртуалке: `touch ~/.ngrok2/ngrok-project.conf && echo web_addr: 0.0.0.0:4040 >> ~/.ngrok2/ngrok-project.conf`
5. Запускаем Ngrok `ngrok http 80 -config=/home/ubuntu/.ngrok2/ngrok-project.conf`. Аналитический интерфес - по адресу сервера + порт 4040.
## Reference:
* [Ngrok](https://ngrok.com/download)
* [Accessing Ngrok interface](https://stackoverflow.com/questions/33358414/accessing-ngrok-web-interface-on-a-vagrant-box)
