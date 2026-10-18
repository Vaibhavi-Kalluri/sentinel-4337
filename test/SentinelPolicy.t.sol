// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {SentinelPolicy} from "../src/SentinelPolicy.sol";

contract SentinelPolicyTest is Test {
    SentinelPolicy policy;

    address target = address(0x1234);

    // Selector for setValue(uint256)
    bytes4 constant SET_VALUE_SELECTOR = 0x55241077;

    function setUp() public {
        policy = new SentinelPolicy();
    }

    function testAllowsApprovedTarget() public {
        policy.setTargetAllowed(target, true);

        bytes memory callData =
            abi.encodeWithSelector(SET_VALUE_SELECTOR, 42);

        (bool allowed, string memory reason) =
            policy.checkOperation(target, callData);

        assertTrue(allowed);
        assertEq(reason, "POLICY_APPROVED");
    }

    function testRejectsUnapprovedTarget() public {
        bytes memory callData =
            abi.encodeWithSelector(SET_VALUE_SELECTOR, 42);

        (bool allowed, string memory reason) =
            policy.checkOperation(target, callData);

        assertFalse(allowed);
        assertEq(reason, "TARGET_NOT_ALLOWED");
    }

    function testRejectsBlockedSelector() public {
        policy.setTargetAllowed(target, true);
        policy.setSelectorBlocked(SET_VALUE_SELECTOR, true);

        bytes memory callData =
            abi.encodeWithSelector(SET_VALUE_SELECTOR, 42);

        (bool allowed, string memory reason) =
            policy.checkOperation(target, callData);

        assertFalse(allowed);
        assertEq(reason, "SELECTOR_BLOCKED");
    }

    function testRejectsShortCalldata() public {
        policy.setTargetAllowed(target, true);

        bytes memory callData = hex"1234";

        (bool allowed, string memory reason) =
            policy.checkOperation(target, callData);

        assertFalse(allowed);
        assertEq(reason, "INVALID_CALLDATA");
    }
}
