import asyncio
from bleak import BleakClient, BleakScanner
import threading


implemented_codecs = ['int16']
def decode_data(data, codec):
    try:
        if codec == 'int16':
            samples = [int.from_bytes(data[i:i+2], byteorder='little', signed=True) for i in range(0, len(data), 2)]
        else:
            raise ValueError(f"Codec {codec} not implemented.")
    except Exception as e:
        print(f"Error decoding data with codec {codec}: {e}")
        return data  # Return raw data on error
    return samples  # Return decoded samples

def encode_data(samples, codec):
    try:
        if codec == 'int16':
            data = bytearray()
            for sample in samples:
                data.extend(sample.to_bytes(2, byteorder='little', signed=True))
        else:
            raise ValueError(f"Codec {codec} not implemented.")
    except Exception as e:
        print(f"Error encoding data with codec {codec}: {e}")
        return bytearray()  # Return empty bytearray on error
    return data  # Return encoded bytearray


class BLEInterface:
            
    def __init__(self):

        
        self.client = None
        self.device = None

        self.device_name = None
        self.device_address = None

        
        
        self.disconnected_event = asyncio.Event()

        # Loop y thread para manejar BLE en un hilo separado al GUI
        self.up = True
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_ble_loop, daemon=True)
        self.thread.start()

    
    def cleanup(self):
        if self.client and self.client.is_connected:
            print("Cleaning up BLE connection...")
            asyncio.run_coroutine_threadsafe(self.client.disconnect(), self.loop).result()
            print("BLE connection closed.")
        self.up = False
        


    def run_ble_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run())


    async def run(self):
        while self.up:
            await asyncio.sleep(0.001)  # Keep BLE alive



    def scan_nearby_devices(self):
        '''
            Discovers nearby BLE devices. returns a list of discovered devices.
            [(device.name, device.address)....]
        '''
        async def inner():
            devices = await BleakScanner.discover()
            print("Discovered devices:", [(device.name, device.address) for device in devices])
            return [(device.name, device.address) for device in devices]
        try:
            fut = asyncio.run_coroutine_threadsafe(inner(), self.loop)
            result = fut.result()
        except RuntimeError:
            # If no event loop is running, create one
            result = asyncio.run(inner())
        return result
    

        
    def connect_by_name(self, name):
        '''
            Connects to a BLE device by its name.
            Returns True if the connection is successful, otherwise returns False.
        '''
        self.device = None
        async def _find_device_by_name(self):
            devices = await BleakScanner.discover()
            print("Discovered devices:", [(device.name, device.address) for device in devices])
            for device in devices:
                if device.name == name:
                    self.device = device
                    return device
            print(f"Device named '{name}' not found")
            return None
        
        async def inner():
            try:
                if not self.device:
                    self.device = await _find_device_by_name(self)
                    if not self.device:
                        print("Device not found.")
                        return False

                self.client = BleakClient(self.device, disconnected_callback=self.on_disconnected)
                await self.client.connect()
                print(f"Connected to {name}")
                # Rediscover services and characteristics to avoid cache issues
                print(self.client.services)
                print("Services rediscovered successfully.")
                return True
            except Exception as e:
                print(f"Connection failed: {e}")
                self.client = None
                return False
        try:
            fut = asyncio.run_coroutine_threadsafe(inner(), self.loop)
            result = fut.result()   
        except RuntimeError:
            # If no event loop is running, create one
            result = asyncio.run(inner())
        return result

    def disconnect(self):
        '''
            Disconnects from the currently connected BLE device.
            Clears the client attribute after disconnection.
            Raises an exception if no device is currently connected.
        '''
        async def inner():
            if self.get_connection_status():
                await self.client.disconnect()
                print(f"Disconnected from {self.device_name}")
                self.client = None
                return
            raise Exception(f"Device named '{self.device_name}' not connected")
        try:
            fut = asyncio.run_coroutine_threadsafe(inner(), self.loop)
            result = fut.result()
        except RuntimeError:
            return asyncio.run(inner())
    
    def get_connection_status(self):
        '''
            Returns the connection status of the BLE device.
            Returns True if the device is connected, otherwise returns False.
        '''
        return not self.client is None and self.client.is_connected
    
    def get_services(self):
        '''
            Retrieves all GATT services from the connected BLE device.
            Raises an exception if no device is connected.
        '''
        USER_DESC_UUID = "00002901-0000-1000-8000-00805f9b34fb"
        async def inner():
            if self.client and self.client.is_connected:
                services = self.client.services
                services_dict = {}
                for service in services:
                    char_info = []
                    for characteristic in service.characteristics:
                        # Try to get the user descriptor value if available
                        user_desc = None
                        for descriptor in characteristic.descriptors:

                            if descriptor.uuid == USER_DESC_UUID:  
                                try:
                                    user_desc = await self.client.read_gatt_descriptor(descriptor.handle)
                                    # Decode bytes to string if possible
                                    try:
                                        user_desc = user_desc.decode('utf-8').strip()
                                    except Exception:
                                        pass
                                except Exception as e:
                                    user_desc = f"Error reading descriptor: {e}"
                        char_info.append({
                            "uuid": characteristic.uuid,
                            "user_description": user_desc
                        })
                    services_dict[service.uuid] = char_info
                return services_dict
            else:
                raise Exception("Device not connected. Please connect first.")
        try:
            fut = asyncio.run_coroutine_threadsafe(inner(), self.loop)
            return fut.result()
        except RuntimeError:
            return asyncio.run(inner())
    
    def get_address(self):
        '''
            Retrieves the MAC address of the connected BLE device.
            Raises an exception if the device is not found or not connected.
        '''
        async def inner():
            if self.device:
                return self.device.address
            else:
                raise Exception("Device not found. Please connect first.")
        try:
            fut = asyncio.run_coroutine_threadsafe(inner(), self.loop)
            return fut.result()
        except RuntimeError:
            return asyncio.run(inner())
        
    def get_name(self):
        '''
            Retrieves the name of the connected BLE device.
            Raises an exception if the device is not found or not connected.
        '''
        async def inner():
            if self.device:
                return self.device.name
            else:
                raise Exception("Device not found. Please connect first.")
        try:
            fut = asyncio.run_coroutine_threadsafe(inner(), self.loop)
            return fut.result()
        except RuntimeError:
            return asyncio.run(inner()  )
        
    def read_characteristic(self, uuid, data_type=None):
        '''
            Reads and returns the value of a specified GATT characteristic from the connected BLE device.
            Raises an exception if no device is connected.
        '''
        async def inner():
            if self.client and self.client.is_connected:
                value = await self.client.read_gatt_char(uuid)
                value = decode_data(value, data_type) if data_type else value
                print(f"Read value from characteristic {uuid}: {value}")
                return value
            else:
                raise Exception("Device not connected. Please connect first.")
        try:
            fut = asyncio.run_coroutine_threadsafe(inner(), self.loop)
            return fut.result()
        except RuntimeError:
            return asyncio.run(inner())
    
    def write_characteristic(self, uuid, value, data_type=None):
        '''
            Writes a specified value to a GATT characteristic on the connected BLE device.
            Raises an exception if no device is connected.
        '''
        async def inner():
            if self.client and self.client.is_connected:
                data = encode_data(value, data_type) if data_type else value
                await self.client.write_gatt_char(uuid, data)
                print(f"Written value to characteristic {uuid}")
            else:
                raise Exception("Device not connected. Please connect first.")
        try:
            fut = asyncio.run_coroutine_threadsafe(inner(), self.loop)
            return fut.result()
        except RuntimeError:
            return asyncio.run(inner()) 
        

    


    def on_disconnected(self, client):
        """
        Callback triggered when the device disconnects.
        """
        print("Device has been disconnected (callback).")
        self.disconnected_event.set()
        



    #--------------------------------------------------------------
    def make_notification_handler(self, char_uuid, callback, codec):
        async def handler(sender, data):
            #print(f"[Async handler] Notification from {char_uuid}, reporting to: {callback.__name__}, data: {data}")
            #print(f"[Async handler] Notification from {char_uuid}, reporting to: {callback.__name__}")
            data = decode_data(data, codec)
            if callback:
                callback(char_uuid, data)
        return handler

    def subscribe_to_char_notifications(self, char_uuid, callback, data_type=None):
        if not self.get_connection_status():
            print("BLE client not ready yet.")
            return
        if data_type not in implemented_codecs and data_type is not None:
            print(f"Warning: Codec {data_type} not recognized. Proceeding without codec.")
            data_type = None
        fut = asyncio.run_coroutine_threadsafe(self._subscribe_coro(char_uuid, callback, data_type), self.loop)
        return fut

    async def _subscribe_coro(self, char_uuid, callback, data_type):
        if not self.get_connection_status():
            print("BLE client not connected.")
            return
        await self.client.start_notify(char_uuid, self.make_notification_handler(char_uuid, callback, data_type))
        print(f"Subscribed to notifications for {char_uuid} (on demand).")

    def unsubscribe_to_char_notifications(self, char_uuid):
        if not self.get_connection_status():
            print("BLE client not ready yet.")
            return
        fut = asyncio.run_coroutine_threadsafe(self._unsubscribe_coro(char_uuid), self.loop)
        return fut

    async def _unsubscribe_coro(self, char_uuid):
        if not self.get_connection_status():
            print("BLE client not connected.")
            return
        await self.client.stop_notify(char_uuid)
        print(f"Unsubscribed from notifications for {char_uuid}.")

