#!/usr/bin/env python3

import CTAFlow
import numpy as np
import pandas as pd
from datetime import datetime

def test_contract_access():
    """Test that both M{i} and month codes work for contract access"""
    
    print("Testing Contract Access Methods")
    print("=" * 50)
    
    # Test with CL_F (oil)
    try:
        spread_data = CTAFlow.SpreadData('CL_F')
        print(f"Loaded SpreadData for CL_F")
        print(f"Available seq_labels: {spread_data.seq_data.seq_labels if spread_data.seq_data else 'None'}")
        
        # Test sequential access (M0, M1, M2)
        print("\nTesting Sequential Access (M0, M1, M2):")
        try:
            contract_m0 = spread_data['M0']
            print(f"M0 access: SUCCESS - {type(contract_m0)} with label '{contract_m0.label}'")
            print(f"  Data shape: {contract_m0.data.shape}")
            print(f"  Continuous: {contract_m0.continuous}")
            print(f"  Is front month: {contract_m0.is_front_month}")
        except Exception as e:
            print(f"M0 access: FAILED - {e}")
            
        try:
            contract_m1 = spread_data['M1'] 
            print(f"M1 access: SUCCESS - {type(contract_m1)} with label '{contract_m1.label}'")
        except Exception as e:
            print(f"M1 access: FAILED - {e}")
            
        # Test month code access (F, G, H, etc.)
        print("\nTesting Month Code Access:")
        if hasattr(spread_data, 'curve_month_labels'):
            available_months = spread_data.curve_month_labels
            print(f"Available month labels: {available_months}")
            
            if available_months:
                first_month = available_months[0]
                try:
                    contract_month = spread_data[first_month]
                    print(f"{first_month} access: SUCCESS - {type(contract_month)} with label '{contract_month.label}'")
                except Exception as e:
                    print(f"{first_month} access: FAILED - {e}")
        else:
            print("No curve_month_labels available")
            
        # Test _get_contract_prices method
        print("\nTesting _get_contract_prices method:")
        test_dates = spread_data.index[:5] if spread_data.index is not None else []
        if len(test_dates) > 0:
            try:
                prices_m0 = spread_data._get_contract_prices('M0', test_dates)
                print(f"_get_contract_prices('M0'): SUCCESS - shape {prices_m0.shape}")
            except Exception as e:
                print(f"_get_contract_prices('M0'): FAILED - {e}")
                
            # Try with month code if available
            if hasattr(spread_data, 'curve_month_labels') and spread_data.curve_month_labels:
                first_month = spread_data.curve_month_labels[0]
                try:
                    prices_month = spread_data._get_contract_prices(first_month, test_dates)
                    print(f"_get_contract_prices('{first_month}'): SUCCESS - shape {prices_month.shape}")
                except Exception as e:
                    print(f"_get_contract_prices('{first_month}'): FAILED - {e}")
        else:
            print("No test dates available")
            
    except Exception as e:
        print(f"Failed to load CL_F: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_contract_access()