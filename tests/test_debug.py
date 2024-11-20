import sys
import os
import traceback
import inspect

def get_call_stack():
    """Get the current call stack information"""
    stack = inspect.stack()
    return [f"{frame.filename}:{frame.lineno} - {frame.function}" for frame in stack[1:]]

def nested_function():
    """Test nested function call"""
    var_in_nested = "I'm in nested function"
    print(f"Local var: {var_in_nested}")
    return get_call_stack()

def test_debug_environment():
    """
    Test function to verify debug environment setup with variables and call stack
    """
    try:
        # Test variables at different scopes
        global_var = "I'm a global variable"
        local_var = "I'm a local variable"
        
        # Create a list and dict for debugging
        test_list = [1, 2, 3, 4, 5]
        test_dict = {"key1": "value1", "key2": "value2"}
        
        # Check Python version
        print(f"Python version: {sys.version}")
        
        # Check if debugger is attached
        debugger_attached = hasattr(sys, 'gettrace') and sys.gettrace() is not None
        print(f"Debugger attached: {debugger_attached}")
        
        # Check PYTHONPATH
        print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        
        # Get call stack from nested function
        call_stack = nested_function()
        print("\nCall stack:")
        for frame in call_stack:
            print(f"  {frame}")
            
        # Print variable information
        print("\nLocal variables:")
        print(f"  global_var = {global_var}")
        print(f"  local_var = {local_var}")
        print(f"  test_list = {test_list}")
        print(f"  test_dict = {test_dict}")
        
        # Try importing project modules
        try:
            import QR1
            print("\nSuccessfully imported QR1 module")
        except ImportError as e:
            print(f"\nFailed to import QR1: {e}")
            print("Traceback:")
            print(traceback.format_exc())
            raise
            
        return True
        
    except Exception as e:
        print(f"Debug environment test failed: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    result = test_debug_environment()
    if result:
        print("Debug environment test passed successfully")
        sys.exit(0)
    else:
        print("Debug environment test failed")
        sys.exit(1)
