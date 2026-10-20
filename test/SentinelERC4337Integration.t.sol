
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {Test} from "forge-std/Test.sol";
import {EntryPoint} from "account-abstraction/core/EntryPoint.sol";
import {PackedUserOperation} from "account-abstraction/interfaces/PackedUserOperation.sol";
import {BaseAccount} from "account-abstraction/core/BaseAccount.sol";
import {SentinelTarget} from "../src/SentinelTarget.sol";
import {SentinelPolicy} from "../src/SentinelPolicy.sol";
import {SentinelAccount} from "../src/SentinelAccount.sol";
import {SentinelAccountFactory} from "../src/SentinelAccountFactory.sol";

contract SentinelERC4337IntegrationTest is Test {
    EntryPoint internal entryPoint;
    SentinelPolicy internal policy;
    SentinelAccountFactory internal factory;
    SentinelTarget internal target;

    uint256 internal ownerPrivateKey;
    address internal owner;
    address internal beneficiary;

    function setUp() public {
        entryPoint = new EntryPoint();
        policy = new SentinelPolicy();
        factory = new SentinelAccountFactory(entryPoint, policy);
        target = new SentinelTarget();

        ownerPrivateKey = 0xA11CE;
        owner = vm.addr(ownerPrivateKey);
        beneficiary = makeAddr("beneficiary");

        vm.txGasPrice(1 gwei);

        // Permit the target used by the successful operation.
        policy.setTargetAllowed(address(target), true);

        // Give EntryPoint funds to cover UserOperation prefunding.
        vm.deal(address(this), 10 ether);
    }

    function _buildUserOp(
        uint256 salt,
        address targetAddress,
        bytes memory innerCall
    ) internal returns (PackedUserOperation memory userOp) {
        address sender = factory.getAddress(owner, salt);

        bytes memory initCode = abi.encodePacked(
            address(factory),
            abi.encodeCall(
                SentinelAccountFactory.createAccount,
                (owner, salt)
            )
        );

        bytes memory callData = abi.encodeWithSignature(
            "execute(address,uint256,bytes)",
            targetAddress,
            0,
            innerCall
        );

        userOp.sender = sender;
        userOp.nonce = entryPoint.getNonce(sender, 0);
        userOp.initCode = initCode;
        userOp.callData = callData;
        userOp.accountGasLimits = bytes32(
            (uint256(1_000_000) << 128) | uint256(500_000)
        );
        userOp.preVerificationGas = 50_000;
        userOp.gasFees = bytes32(
            (uint256(1 gwei) << 128) | uint256(2 gwei)
        );
        userOp.paymasterAndData = "";

        bytes32 userOpHash = entryPoint.getUserOpHash(userOp);
        (uint8 v, bytes32 r, bytes32 s) =
            vm.sign(ownerPrivateKey, userOpHash);
        userOp.signature = abi.encodePacked(r, s, v);
    }

    function _submit(PackedUserOperation memory userOp) internal {
        PackedUserOperation[] memory ops =
            new PackedUserOperation[](1);
        ops[0] = userOp;

        vm.startPrank(owner, owner);
        entryPoint.handleOps(ops, payable(beneficiary));
        vm.stopPrank();
    }

    function testAllowedOperationExecutesThroughEntryPoint() public {
        uint256 salt = 1;

        address sender = factory.getAddress(owner, salt);

        // Prefund the counterfactual account.
        entryPoint.depositTo{value: 1 ether}(sender);

        PackedUserOperation memory userOp = _buildUserOp(
            salt,
            address(target),
            abi.encodeCall(SentinelTarget.setValue, (42))
        );

        _submit(userOp);

        assertGt(sender.code.length, 0);
        assertEq(target.value(), 42);
    }

    function testForbiddenTargetDoesNotExecuteThroughEntryPoint() public {
        uint256 salt = 2;
        SentinelTarget forbiddenTarget = new SentinelTarget();

        address sender = factory.getAddress(owner, salt);
        entryPoint.depositTo{value: 1 ether}(sender);

        PackedUserOperation memory userOp = _buildUserOp(
            salt,
            address(forbiddenTarget),
            abi.encodeCall(SentinelTarget.setValue, (99))
        );

        // EntryPoint may handle account execution failure without
        // reverting the whole handleOps transaction.
        _submit(userOp);

        assertGt(sender.code.length, 0);
        assertEq(forbiddenTarget.value(), 0);
    }
function testBlockedSelectorDoesNotExecuteThroughEntryPoint() public {
    uint256 salt = 3;
    address sender = factory.getAddress(owner, salt);

    entryPoint.depositTo{value: 1 ether}(sender);

    // Allow the target, but block setValue(uint256).
    policy.setSelectorBlocked(
        SentinelTarget.setValue.selector,
        true
    );

    PackedUserOperation memory userOp = _buildUserOp(
        salt,
        address(target),
        abi.encodeCall(SentinelTarget.setValue, (99))
    );

    _submit(userOp);

    assertGt(sender.code.length, 0);
    assertEq(target.value(), 0);
}

function testBatchValidatesAllCallsBeforeExecuting() public {
    uint256 salt = 4;
    address sender = factory.getAddress(owner, salt);

    entryPoint.depositTo{value: 1 ether}(sender);

    SentinelTarget forbiddenTarget = new SentinelTarget();

    // First call is allowed; second call targets a forbidden target.
    BaseAccount.Call[] memory calls =
    new BaseAccount.Call[](2);
    calls[0] = BaseAccount.Call({
        target: address(target),
        value: 0,
        data: abi.encodeCall(SentinelTarget.setValue, (77))
    });

    calls[1] = BaseAccount.Call({
        target: address(forbiddenTarget),
        value: 0,
        data: abi.encodeCall(SentinelTarget.setValue, (99))
    });

    bytes memory callData = abi.encodeWithSignature(
        "executeBatch((address,uint256,bytes)[])",
        calls
    );

    PackedUserOperation memory userOp = _buildUserOp(
        salt,
        address(target),
        abi.encodeCall(SentinelTarget.setValue, (77))
    );

    // Replace the single-call calldata with executeBatch calldata.
    userOp.callData = callData;

    // Re-sign because changing callData changes the UserOperation hash.
    bytes32 userOpHash = entryPoint.getUserOpHash(userOp);
    (uint8 v, bytes32 r, bytes32 s) =
        vm.sign(ownerPrivateKey, userOpHash);
    userOp.signature = abi.encodePacked(r, s, v);

    _submit(userOp);

    // The allowed first call must not execute if the second is forbidden.
    assertEq(target.value(), 0);
    assertEq(forbiddenTarget.value(), 0);
}
}
