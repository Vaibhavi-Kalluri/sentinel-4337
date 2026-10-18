
 // SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {Test} from "forge-std/Test.sol";
import {EntryPoint} from "account-abstraction/core/EntryPoint.sol";
import {BaseAccount} from "account-abstraction/core/BaseAccount.sol";
import {IEntryPoint} from "account-abstraction/interfaces/IEntryPoint.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

import {SentinelAccount} from "../src/SentinelAccount.sol";
import {SentinelPolicy} from "../src/SentinelPolicy.sol";
import {SentinelTarget} from "../src/SentinelTarget.sol";

contract SentinelAccountTest is Test {
    EntryPoint internal entryPoint;
    SentinelPolicy internal policy;
    SentinelAccount internal account;
    SentinelTarget internal target;

    address internal owner = address(this);
    address internal forbiddenTarget = address(0xBEEF);

    function setUp() public {
        entryPoint = new EntryPoint();
        policy = new SentinelPolicy();
        target = new SentinelTarget();

        SentinelAccount implementation =
            new SentinelAccount(IEntryPoint(address(entryPoint)), policy);

        ERC1967Proxy proxy = new ERC1967Proxy(
            address(implementation),
            abi.encodeWithSignature("initialize(address)", owner)
        );

        account = SentinelAccount(payable(address(proxy)));

        // Allow the test target by default.
        policy.setTargetAllowed(address(target), true);
    }

    function testAllowedCallExecutes() public {
        bytes memory callData =
            abi.encodeWithSelector(SentinelTarget.setValue.selector, 42);

        account.execute(address(target), 0, callData);

        assertEq(target.value(), 42);
    }

    function testUnapprovedTargetIsRejected() public {
        bytes memory callData =
            abi.encodeWithSelector(SentinelTarget.setValue.selector, 42);

        vm.expectRevert(
            abi.encodeWithSelector(
                SentinelAccount.OperationBlocked.selector,
                forbiddenTarget,
                "TARGET_NOT_ALLOWED"
            )
        );

        account.execute(forbiddenTarget, 0, callData);
    }

    function testBlockedSelectorIsRejected() public {
        bytes4 selector = SentinelTarget.setValue.selector;
        policy.setSelectorBlocked(selector, true);

        bytes memory callData =
            abi.encodeWithSelector(selector, 42);

        vm.expectRevert(
            abi.encodeWithSelector(
                SentinelAccount.OperationBlocked.selector,
                address(target),
                "SELECTOR_BLOCKED"
            )
        );

        account.execute(address(target), 0, callData);

        assertEq(target.value(), 0);
    }

    function testBatchRejectsBeforeAnyCallExecutes() public {
        SentinelTarget secondTarget = new SentinelTarget();

        // The first target is allowed; the second is not.
        policy.setTargetAllowed(address(secondTarget), false);

        BaseAccount.Call[] memory calls =
            new BaseAccount.Call[](2);

        calls[0] = BaseAccount.Call({
            target: address(target),
            value: 0,
            data: abi.encodeWithSelector(
                SentinelTarget.setValue.selector, 100
            )
        });

        calls[1] = BaseAccount.Call({
            target: address(secondTarget),
            value: 0,
            data: abi.encodeWithSelector(
                SentinelTarget.setValue.selector, 200
            )
        });

        vm.expectRevert(
            abi.encodeWithSelector(
                SentinelAccount.OperationBlocked.selector,
                address(secondTarget),
                "TARGET_NOT_ALLOWED"
            )
        );

        account.executeBatch(calls);

        // The first call must not have executed either.
        assertEq(target.value(), 0);
        assertEq(secondTarget.value(), 0);
    }
}
