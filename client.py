import asyncio
import ssl
import websockets
import time

TOKEN = "my_secret_token"

async def send_data():
    uri = "wss://205.172.57.158:8765"

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE


    async with websockets.connect(uri, ssl=ssl_context) as websocket:
        for i in range(100000):
            end_time_first = time.time()
            await websocket.send(TOKEN)
            auth_response = await websocket.recv()
            print(f"Ответ сервера: {auth_response}")
            if "не удалась" in auth_response.lower():
                return

            data_to_process = "Это данные для обработки"
            await websocket.send(data_to_process)

            while True:
                response = await websocket.recv()
                end_time_second = time.time()
                time_difference = end_time_second - end_time_first
                print(f"Время между двумя событиями: {time_difference} секунд")
                i = 0
                if response == "Обработка завершена":
                    print("Получены все чанки.")
                    break
                else:
                    print(f"Получен чанк от сервера: {response}")
                    assert i < int(response[5:7])
                    i=int(response[5:7])

asyncio.get_event_loop().run_until_complete(send_data())
