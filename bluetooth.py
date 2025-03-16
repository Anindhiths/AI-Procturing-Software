#!/usr/bin/env python3
import objc
from Foundation import NSObject, NSArray
import CoreBluetooth
import time
import subprocess
import re
import json
import os
import plistlib

class BluetoothDelegate(NSObject):
    def init(self):
        self = objc.super(BluetoothDelegate, self).init()
        if self is None:
            return None
        self.discovered_devices = {}
        self.finished = False
        return self
    
    def centralManagerDidUpdateState_(self, central):
        states = {
            CoreBluetooth.CBManagerStateUnknown: "Unknown",
            CoreBluetooth.CBManagerStateResetting: "Resetting",
            CoreBluetooth.CBManagerStateUnsupported: "Unsupported",
            CoreBluetooth.CBManagerStateUnauthorized: "Unauthorized",
            CoreBluetooth.CBManagerStatePoweredOff: "Powered Off",
            CoreBluetooth.CBManagerStatePoweredOn: "Powered On"
        }
        
        print(f"Bluetooth status: {states.get(central.state(), 'Unknown')}")
        
        if central.state() == CoreBluetooth.CBManagerStatePoweredOn:
            print("Scanning for BLE devices...")
            # Scan with options to detect more devices
            options = {CoreBluetooth.CBCentralManagerScanOptionAllowDuplicatesKey: True}
            central.scanForPeripheralsWithServices_options_(None, options)
    
    def centralManager_didDiscoverPeripheral_advertisementData_RSSI_(self, central, peripheral, data, rssi):
        name = peripheral.name() or "Unknown Device"
        uuid = peripheral.identifier().UUIDString()
        
        # Check for connected state
        state = "Discovered"
        if peripheral.state() == CoreBluetooth.CBPeripheralStateConnected:
            state = "Connected"
        elif peripheral.state() == CoreBluetooth.CBPeripheralStateDisconnected:
            state = "Disconnected"
        
        self.discovered_devices[uuid] = {
            'name': name,
            'uuid': uuid,
            'rssi': rssi,
            'state': state,
            'type': 'BLE',
            'advertisement_data': str(data)
        }
        #print(f"Found BLE device: {name} ({uuid}) - {state} - RSSI: {rssi}")
    
    def stop_scan(self, central):
        central.stopScan()
        self.finished = True
        print("BLE scan completed")


def scan_ble_devices(duration=30):
    """
    Scan for Bluetooth Low Energy devices for the specified duration
    
    Args:
        duration (int): Scan duration in seconds
        
    Returns:
        dict: Dictionary of discovered devices
    """
    # Create delegate and central manager
    delegate = BluetoothDelegate.alloc().init()
    central_manager = CoreBluetooth.CBCentralManager.alloc().initWithDelegate_queue_(delegate, None)
    
    # Run the scan for the specified duration
    print(f"Scanning for {duration} seconds...")
    timeout = time.time() + duration
    while time.time() < timeout and not delegate.finished:
        time.sleep(0.1)
    
    # Stop scanning
    delegate.stop_scan(central_manager)
    
    return delegate.discovered_devices


def get_paired_devices_blueutil():
    """
    Get paired devices using blueutil
    Install blueutil with: brew install blueutil
    
    Returns:
        list: List of paired devices
    """
    try:
        # Check if blueutil is installed
        check_blueutil = subprocess.run(["which", "blueutil"], capture_output=True, text=True)
        if check_blueutil.returncode != 0:
            print("blueutil not found. Install with: brew install blueutil")
            return []
        
        # Get paired devices
        result = subprocess.run(["blueutil", "--paired"], capture_output=True, text=True)
        
        devices = []
        for line in result.stdout.splitlines():
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2:
                    address = parts[0].strip()
                    name = parts[1].strip()
                    
                    # Get connection status
                    connected = "Unknown"
                    info_result = subprocess.run(["blueutil", "--info", address], 
                                                capture_output=True, text=True)
                    if "connected: 1" in info_result.stdout:
                        connected = "Connected"
                    elif "connected: 0" in info_result.stdout:
                        connected = "Disconnected"
                    
                    devices.append({
                        'name': name,
                        'address': address,
                        'connected': connected,
                        'type': 'Classic/BLE',
                        'source': 'blueutil'
                    })
        
        return devices
    except Exception as e:
        print(f"Error getting paired devices with blueutil: {e}")
        return []


def get_system_profiler_devices():
    """
    Get Bluetooth devices using system_profiler
    
    Returns:
        list: List of Bluetooth devices
    """
    try:
        # Run system profiler with full detail level
        result = subprocess.run(
            ["system_profiler", "SPBluetoothDataType", "-json"], 
            capture_output=True, 
            text=True
        )
        
        devices = []
        
        try:
            # Parse JSON output
            data = json.loads(result.stdout)
            
            # Navigate the nested structure
            bluetooth_data = data.get('SPBluetoothDataType', [{}])[0]
            
            # Get device information
            device_sections = ["device_connected", "device_not_connected", "devices_other"]
            
            for section in device_sections:
                device_dict = bluetooth_data.get(section, {})
                if not device_dict:
                    continue
                
                for device_key, device_info in device_dict.items():
                    name = device_key
                    connected = "Connected" if section == "device_connected" else "Disconnected"
                    address = device_info.get('device_address', 'Unknown')
                    device_type = device_info.get('device_type', 'Unknown')
                    
                    devices.append({
                        'name': name,
                        'address': address,
                        'connected': connected,
                        'type': device_type,
                        'source': 'system_profiler'
                    })
            
            # Check if no devices were found in the JSON structure
            if not devices:
                # Fallback to text parsing
                text_result = subprocess.run(
                    ["system_profiler", "SPBluetoothDataType"], 
                    capture_output=True, 
                    text=True
                )
                
                output = text_result.stdout
                device_sections = output.split("\n\n")
                
                for section in device_sections:
                    # Skip sections that don't contain device information
                    if not section.strip() or ":" not in section:
                        continue
                    
                    # Try to extract device name, address and connected status
                    lines = section.strip().split("\n")
                    if not lines:
                        continue
                    
                    name = lines[0].split(":")[0].strip()
                    address = "Unknown"
                    connected = "Unknown"
                    
                    for line in lines:
                        if "address:" in line.lower():
                            address = line.split(":", 1)[1].strip()
                        if "connected:" in line.lower():
                            status = line.split(":", 1)[1].strip().lower()
                            connected = "Connected" if status == "yes" else "Disconnected"
                    
                    if name and name != "Bluetooth":
                        devices.append({
                            'name': name,
                            'address': address,
                            'connected': connected,
                            'type': 'Unknown',
                            'source': 'system_profiler_text'
                        })
                
        except json.JSONDecodeError:
            print("Failed to parse system_profiler JSON output, falling back to text parsing...")
            # Fallback to text parsing (similar to above)
            text_result = subprocess.run(
                ["system_profiler", "SPBluetoothDataType"], 
                capture_output=True, 
                text=True
            )
            
            # Simple text parsing for most cases
            output = text_result.stdout
            in_devices_section = False
            current_device = {}
            
            for line in output.splitlines():
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Look for device name at start of new block
                if not line.startswith(" ") and ":" in line:
                    # Save previous device if it exists
                    if current_device.get('name') and current_device.get('name') != "Bluetooth":
                        devices.append(current_device)
                    
                    # Start new device
                    current_device = {
                        'name': line.split(":", 1)[0].strip(),
                        'address': 'Unknown',
                        'connected': 'Unknown',
                        'type': 'Unknown',
                        'source': 'system_profiler_text'
                    }
                
                # Look for device address
                elif "address:" in line.lower():
                    current_device['address'] = line.split(":", 1)[1].strip()
                
                # Look for connection status
                elif "connected:" in line.lower():
                    status = line.split(":", 1)[1].strip().lower()
                    current_device['connected'] = "Connected" if status == "yes" else "Disconnected"
            
            # Add the last device if it exists
            if current_device.get('name') and current_device.get('name') != "Bluetooth":
                devices.append(current_device)
        
        return devices
    
    except Exception as e:
        print(f"Error getting system profiler devices: {e}")
        return []


def get_ioregistry_devices():
    """
    Get Bluetooth devices from IORegistry
    
    Returns:
        list: List of Bluetooth devices
    """
    try:
        # Run ioreg command to get Bluetooth devices
        result = subprocess.run(
            ["ioreg", "-l", "-p", "IOService", "-r", "-c", "IOBluetoothDevice"], 
            capture_output=True, 
            text=True
        )
        
        devices = []
        current_device = {}
        
        for line in result.stdout.splitlines():
            line = line.strip()
            
            # New device block starts
            if "IOBluetoothDevice" in line:
                if current_device.get('name'):
                    devices.append(current_device)
                current_device = {
                    'name': 'Unknown',
                    'address': 'Unknown',
                    'connected': 'Unknown',
                    'type': 'IORegistry',
                    'source': 'ioreg'
                }
            
            # Device name
            if "\"device name\"" in line:
                match = re.search(r'"device name" = "([^"]+)"', line)
                if match:
                    current_device['name'] = match.group(1)
            
            # Device address
            if "\"device address\"" in line:
                match = re.search(r'"device address" = "([^"]+)"', line)
                if match:
                    current_device['address'] = match.group(1)
            
            # Connected status
            if "\"connected\"" in line:
                match = re.search(r'"connected" = (\w+)', line)
                if match and match.group(1) == "Yes":
                    current_device['connected'] = "Connected"
                else:
                    current_device['connected'] = "Disconnected"
        
        # Add the last device
        if current_device.get('name'):
            devices.append(current_device)
        
        return devices
    
    except Exception as e:
        print(f"Error getting IORegistry devices: {e}")
        return []


def get_bluetooth_plist_devices():
    """
    Get devices from Bluetooth preferences plist file
    
    Returns:
        list: List of devices
    """
    try:
        # Path to the Bluetooth preferences plist
        plist_path = "/Library/Preferences/com.apple.Bluetooth.plist"
        
        if not os.path.exists(plist_path):
            print(f"Bluetooth plist not found at {plist_path}")
            return []
        
        # Try to read the plist file
        try:
            # First attempt using plistlib
            with open(plist_path, 'rb') as file:
                plist_data = plistlib.load(file)
        except:
            # Fallback to using defaults command
            result = subprocess.run(
                ["defaults", "read", "/Library/Preferences/com.apple.Bluetooth"], 
                capture_output=True, 
                text=True
            )
            
            # Can't easily parse this output in a reliable way
            return []
        
        devices = []
        
        # Try to extract device information
        # The structure can vary between macOS versions
        paired_devices = plist_data.get('PairedDevices', [])
        
        for device_address in paired_devices:
            # Convert binary address to string
            if isinstance(device_address, bytes):
                addr_str = ':'.join(f'{b:02x}'.upper() for b in device_address)
            else:
                addr_str = str(device_address)
            
            devices.append({
                'name': 'Unknown (from plist)',
                'address': addr_str,
                'connected': 'Unknown',
                'type': 'Paired',
                'source': 'plist'
            })
        
        return devices
    
    except Exception as e:
        print(f"Error getting devices from plist: {e}")
        return []


def merge_device_data(all_devices):
    """
    Merge device data from different sources to remove duplicates
    
    Args:
        all_devices (list): List of devices from different sources
        
    Returns:
        list: List of unique devices with combined information
    """
    merged_devices = {}
    
    for device in all_devices:
        # Use address as key for merging
        key = device.get('address', '').lower()
        if not key or key == 'unknown':
            # If no address, try using UUID
            key = device.get('uuid', '').lower()
        
        if not key or key == 'unknown':
            # If neither address nor UUID, use name
            key = device.get('name', '').lower()
        
        if not key or key == 'unknown':
            # Skip devices with no identifiable information
            continue
        
        if key in merged_devices:
            # Update existing device with any new information
            existing = merged_devices[key]
            
            # Prefer non-unknown values
            if device.get('name') and existing.get('name', 'Unknown') == 'Unknown':
                existing['name'] = device['name']
            
            if device.get('address') and existing.get('address', 'Unknown') == 'Unknown':
                existing['address'] = device['address']
            
            if device.get('uuid') and existing.get('uuid', 'Unknown') == 'Unknown':
                existing['uuid'] = device['uuid']
            
            # Prefer "Connected" status
            # Prefer "Connected" status
            if device.get('connected') == 'Connected':
                existing['connected'] = 'Connected'
            elif device.get('connected') and existing.get('connected', 'Unknown') == 'Unknown':
                existing['connected'] = device['connected']
            
            # Combine sources
            existing['source'] = f"{existing.get('source', '')}+{device.get('source', '')}"
            
            # Keep the most specific type
            if device.get('type') != 'Unknown' and existing.get('type', 'Unknown') == 'Unknown':
                existing['type'] = device['type']
            
            # Keep RSSI if available
            if 'rssi' in device and 'rssi' not in existing:
                existing['rssi'] = device['rssi']
        else:
            # Add new device
            merged_devices[key] = device.copy()
    
    return list(merged_devices.values())


def main():
    print("=== Enhanced Mac Bluetooth Device Scanner ===")
    
    all_devices = []
    
    # Try blueutil first (most reliable for connected devices)
    print("\n1. Checking paired devices with blueutil...")
    blueutil_devices = get_paired_devices_blueutil()
    if blueutil_devices:
        print(f"Found {len(blueutil_devices)} devices with blueutil")
        all_devices.extend(blueutil_devices)
    else:
        print("No devices found with blueutil or blueutil not installed")
    
    # Try system profiler
    print("\n2. Checking devices with system_profiler...")
    system_devices = get_system_profiler_devices()
    if system_devices:
        print(f"Found {len(system_devices)} devices with system_profiler")
        all_devices.extend(system_devices)
    else:
        print("No devices found with system_profiler")
    
    # Try IORegistry
    print("\n3. Checking devices with IORegistry...")
    ioreg_devices = get_ioregistry_devices()
    if ioreg_devices:
        print(f"Found {len(ioreg_devices)} devices with IORegistry")
        all_devices.extend(ioreg_devices)
    else:
        print("No devices found with IORegistry")
    
    # Try Bluetooth plist
    print("\n4. Checking devices from Bluetooth preferences...")
    plist_devices = get_bluetooth_plist_devices()
    if plist_devices:
        print(f"Found {len(plist_devices)} devices from Bluetooth preferences")
        all_devices.extend(plist_devices)
    else:
        print("No devices found from Bluetooth preferences")
    
    # Scan for BLE devices
    print("\n5. Scanning for BLE devices...")
    ble_devices = scan_ble_devices(15)
    if ble_devices:
        print(f"Found {len(ble_devices)} BLE devices")
        all_devices.extend(ble_devices.values())
    else:
        print("No BLE devices found or Bluetooth not enabled")
    
    # Merge all device data to remove duplicates
    merged_devices = merge_device_data(all_devices)
    
    # Print results
    print("\n=== Detected Bluetooth Devices ===")
    if merged_devices:
        for i, device in enumerate(merged_devices, 1):
            print(f"\nDevice {i}:")
            print(f"Name: {device.get('name', 'Unknown')}")
            
            if 'address' in device and device['address'] != 'Unknown':
                print(f"Address: {device['address']}")
            
            if 'uuid' in device:
                print(f"UUID: {device['uuid']}")
            
            print(f"Connection Status: {device.get('connected', 'Unknown')}")
            
            if 'rssi' in device:
                print(f"Signal Strength (RSSI): {device['rssi']} dBm")
            
            print(f"Device Type: {device.get('type', 'Unknown')}")
            print(f"Detection Method: {device.get('source', 'Unknown')}")
            print("-" * 40)
        
        print(f"\nTotal unique devices found: {len(merged_devices)}")
    else:
        print("No Bluetooth devices detected")
    
    print("\nNote: This script combines multiple detection methods.")
    print("Some methods require administrator privileges or special permissions.")
    print("If devices are missing, try running with sudo or checking Bluetooth permissions.")


if __name__ == "__main__":
    main()
